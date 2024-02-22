import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import get_embedder
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from .refine_nets.pix2pix import Pix2PixDecoder
from .refine_nets.mlp import RefineMLPDecoder
from .refine_nets.unet import RefineUnetDecoder



class DeformLayerNetwork(nn.Module):
    def __init__(self, img_h=512, img_w=512, t_dim=1, t_multires=0, feature_dim=64, apply_deform=False, parser_type='mlp', layer_parser_rescale=1, out_channel=3):
        super(DeformLayerNetwork, self).__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.feature_dim = feature_dim
        self.embed_time_fn, time_input_ch = get_embedder(t_multires, t_dim)
        self.apply_deform = apply_deform
        assert not apply_deform, "not supported yet"

        self.parser_type = parser_type
        if parser_type == 'pix2pix':
            self.network = Pix2PixDecoder(input_nc=feature_dim + time_input_ch, n_blocks=2, out_rescale=layer_parser_rescale, output_nc=out_channel).cuda()
        elif parser_type == 'mlp':
            self.network = RefineMLPDecoder(
                input_ch=feature_dim + time_input_ch, 
                out_rescale=layer_parser_rescale,
                output_ch=out_channel,
                D=4,
                W=32,
                skips=[],
                act_fn=torch.sigmoid,
                ).cuda()
        elif parser_type == 'unet':
            self.network = RefineUnetDecoder(in_channels=feature_dim + time_input_ch, out_rescale=layer_parser_rescale, out_channels=out_channel).cuda()
        elif parser_type == 'stylegan2':
            raise NotImplementedError(f'refine_parser_type {parser_type} not supported')
            self.network = StyleGan2Gen(
                z_dim=t_dim,
                c_dim=0,
                w_dim=128,
                img_resolution=im_res,
                img_channels=3,
                mapping_kwargs={
                    'num_layers': 2,
                },
                synthesis_kwargs={
                    'channel_base': 32768,
                    'channel_max': 512,
                    'num_fp16_res': 4,
                    'conv_clamp': 256
                }
            )
        else:
            raise NotImplementedError(f'refine_parser_type {parser_type} not supported')
        
        self.feature_img = nn.Parameter(torch.rand([feature_dim, img_h, img_w], device='cuda').requires_grad_(True))


    def forward(self, t):
        # fetch time embedding
        t_emb = self.embed_time_fn(t)
        x = torch.concat([self.feature_img, t_emb.T.unsqueeze(1).repeat([1, self.img_h, self.img_w])], axis=0)
        
        return self.network(x.unsqueeze(0)).squeeze(0)


class LayerModel:
    def __init__(self, layer_model, img_size, t_dim=1, t_multires=0, feature_dim=64, apply_deform=False, parser_type='mlp', layer_parser_rescale=1):
        self.layer_model = layer_model
        # self.bg = nn.Parameter(torch.rand([3, *img_size]).requires_grad_(True)).cuda()
        self.bg_model = DeformLayerNetwork(
            img_size[0], 
            img_size[1], 
            t_dim=t_dim, 
            t_multires=t_multires, 
            feature_dim=feature_dim, 
            apply_deform=apply_deform, 
            parser_type=parser_type, 
            layer_parser_rescale=layer_parser_rescale, 
            out_channel=3,
            )

        if layer_model == 'both':
            self.fg_model = DeformLayerNetwork(
                img_size[0], 
                img_size[1], 
                t_dim=t_dim, 
                t_multires=t_multires, 
                feature_dim=feature_dim, 
                apply_deform=apply_deform, 
                parser_type=parser_type, 
                layer_parser_rescale=layer_parser_rescale, 
                out_channel=4,
                )
        else:
            self.fg_model = None

        self.optimizer = None

    
    def get_background(self, time_emb):
        bg = self.bg_model(time_emb)
        return torch.clamp(bg, 0, 1)
    
    def get_foreground(self, time_emb):
        fg = self.fg_model(time_emb)
        return torch.vstack([torch.clamp(fg[:3], 0, 1), torch.sigmoid(fg[3:])])

    def train_setting(self, training_args):
        l = [
            {'params': list(self.bg_model.parameters()),
             'lr': training_args.layer_bg_lr_init,
             "name": "layer_bg"}
        ]
        if self.fg_model is not None:
            l.append(
            {'params': list(self.fg_model.parameters()),
             'lr': training_args.layer_bg_lr_init,
             "name": "layer_fg"}
            )
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.layer_bg_scheduler_args = get_expon_lr_func(lr_init=training_args.layer_bg_lr_init,
                                                       lr_final=training_args.layer_bg_lr_final,
                                                       lr_delay_mult=training_args.layer_bg_lr_delay_mult,
                                                       max_steps=training_args.layer_bg_lr_max_steps)
        if self.fg_model is not None:
            self.layer_fg_scheduler_args = get_expon_lr_func(lr_init=training_args.layer_fg_lr_init,
                                                        lr_final=training_args.layer_fg_lr_final,
                                                        lr_delay_mult=training_args.layer_fg_lr_delay_mult,
                                                        max_steps=training_args.layer_fg_lr_max_steps)
        


    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "layer/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.bg_model.state_dict(), os.path.join(out_weights_path, 'bg_layer.pth'))
        if self.fg_model is not None:
            torch.save(self.fg_model.state_dict(), os.path.join(out_weights_path, 'fg_layer.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "layer"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "layer/iteration_{}/bg_layer.pth".format(loaded_iter))
        self.bg_model.load_state_dict(torch.load(weights_path))

        if self.fg_model is not None:
            weights_path = os.path.join(model_path, "layer/iteration_{}/fg_layer.pth".format(loaded_iter))
            self.fg_model.load_state_dict(torch.load(weights_path))


    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "layer_bg":
                lr = self.layer_bg_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "layer_fg":
                lr = self.layer_fg_scheduler_args(iteration)
                param_group['lr'] = lr
