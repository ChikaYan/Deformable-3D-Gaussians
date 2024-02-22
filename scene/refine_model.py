import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from utils.time_utils import get_embedder
from utils.stylegan2.networks import Generator as StyleGan2Gen
from .refine_nets.pix2pix import Pix2PixDecoder
from .refine_nets.mlp import RefineMLPDecoder
from .refine_nets.unet import RefineUnetDecoder


class RefineModel:
    def __init__(self, t_dim=1, feature_dim=256, t_multires=0, n_blocks=2, out_rescale=0.001, parser_type='pix2pix', im_res=(512,512)):
        self.t_multires = t_multires
        self.embed_time_fn, time_input_ch = get_embedder(t_multires, t_dim)
        self.optimizer = None

        self.parser_type = parser_type
        if parser_type == 'pix2pix':
            self.network = Pix2PixDecoder(input_nc=feature_dim + time_input_ch, n_blocks=n_blocks, out_rescale=out_rescale).cuda()
        elif parser_type == 'mlp':
            self.network = RefineMLPDecoder(input_ch=feature_dim + time_input_ch, out_rescale=out_rescale).cuda()
        elif parser_type == 'unet':
            self.network = RefineUnetDecoder(in_channels=feature_dim + time_input_ch, out_rescale=out_rescale).cuda()
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

    def step(self, x, t):
        # fetch time embedding
        t_emb = self.embed_time_fn(t)
        _, h, w = x.shape
        x = torch.concat([x, t_emb.T.unsqueeze(1).repeat([1, h, w])], axis=0)
        
        return self.network(x.unsqueeze(0))

    def train_setting(self, training_args):
        l = [
            {'params': list(self.network.parameters()),
             'lr': training_args.refine_lr_init,
             "name": "refine"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.scheduler_args = get_expon_lr_func(lr_init=training_args.refine_lr_init,
                                                       lr_final=training_args.refine_lr_final,
                                                       lr_delay_mult=training_args.refine_lr_delay_mult,
                                                       max_steps=training_args.refine_lr_max_steps)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "refine/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.network.state_dict(), os.path.join(out_weights_path, 'refine.pth'))

    def load_weights(self, model_path, iteration=-1):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "refine"))
        else:
            loaded_iter = iteration
        weights_path = os.path.join(model_path, "refine/iteration_{}/refine.pth".format(loaded_iter))
        self.network.load_state_dict(torch.load(weights_path))

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "refine":
                lr = self.scheduler_args(iteration)
                param_group['lr'] = lr
                return lr
