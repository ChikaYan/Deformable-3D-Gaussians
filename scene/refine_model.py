import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func
from utils.time_utils import get_embedder
from .refine_nets.stylegan2.networks import StyleGan2Gen
from .refine_nets.pix2pix import Pix2PixDecoder
from .refine_nets.mlp import RefineMLPDecoder
from .refine_nets.unet import RefineUnetDecoder
from typing import Literal


class RefineModel:
    def __init__(
            self,
            exp_dim=1,
            feature_dim=256,
            exp_multires=0,
            time_multires=4,
            pix2pix_n_blocks=2,
            out_rescale=0.001,
            parser_type='pix2pix',
            im_res=(512,512),
            stylegan_n_blocks=4,
            stylegan_noise:Literal['random', 'const', 'none', 'time']='random',
            ):
        self.exp_multires = exp_multires
        self.embed_exp_fn, exp_input_ch = get_embedder(exp_multires, exp_dim)
        self.optimizer = None
        self.embed_time_fn, self.time_input_ch = get_embedder(time_multires, 1)

        self.parser_type = parser_type
        if parser_type == 'pix2pix':
            self.network = Pix2PixDecoder(input_nc=feature_dim + exp_input_ch, n_blocks=pix2pix_n_blocks, out_rescale=out_rescale).cuda()
        elif parser_type == 'mlp':
            self.network = RefineMLPDecoder(input_ch=feature_dim + exp_input_ch, out_rescale=out_rescale).cuda()
        elif parser_type == 'unet':
            self.network = RefineUnetDecoder(in_channels=feature_dim + exp_input_ch, out_rescale=out_rescale).cuda()
        elif parser_type == 'stylegan2':
            # raise NotImplementedError(f'refine_parser_type {parser_type} not supported')
            self.network = StyleGan2Gen(
                w_dim=exp_dim, # we directly use exp as w, as it is assumed that exp latent is already well-decoupled
                img_resolution=im_res[0],
                img_channels=3,
                synthesis_kwargs={
                    # 'num_fp16_res': 4,
                    'conv_clamp': 256,
                    'num_blocks': stylegan_n_blocks,
                },
                t_embed_dim=self.time_input_ch,
                noise_mode=stylegan_noise,
            ).cuda()
            self.stylegan_noise = stylegan_noise
        else:
            raise NotImplementedError(f'refine_parser_type {parser_type} not supported')

    def step(self, x, exp, time=None):

        if self.parser_type == 'stylegan2':
            # do not embed exp for stylegan
            time_emb = self.embed_time_fn(time)
            return self.network(x.unsqueeze(0), exp, time_emb, noise_mode=self.stylegan_noise)
        
        else:
            # fetch exp embedding
            exp_emb = self.embed_exp_fn(exp)
            _, h, w = x.shape
            x = torch.concat([x, exp_emb.T.unsqueeze(1).repeat([1, h, w])], axis=0)
            
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
