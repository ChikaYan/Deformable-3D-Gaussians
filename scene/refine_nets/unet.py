import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Tuple, Literal
import numpy as np


class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resample: Literal['default', 'up', 'down'] = 'default',
        groups: int = 32,
        eps: float = 1e-5,
        skip_scale: float = 1, # multiplied to output
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_scale = skip_scale

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.act = F.silu

        self.resample = None
        if resample == 'up':
            self.resample = partial(F.interpolate, scale_factor=2.0, mode="nearest")
        elif resample == 'down':
            self.resample = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.shortcut = nn.Identity()
        if self.in_channels != self.out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

    
    def forward(self, x):
        res = x

        x = self.norm1(x)
        x = self.act(x)

        if self.resample:
            res = self.resample(res)
            x = self.resample(x)
        
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x = (x + self.shortcut(res)) * self.skip_scale

        return x



class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int = 1,
        downsample: bool = True,
        skip_scale: float = 1,
    ):
        super().__init__()
 
        nets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            nets.append(ResnetBlock(in_channels, out_channels, skip_scale=skip_scale))
        self.nets = nn.ModuleList(nets)

        self.downsample = None
        if downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        xs = []

        for net in self.nets:
            x = net(x)
            xs.append(x)

        if self.downsample:
            x = self.downsample(x)
            xs.append(x)
  
        return x, xs


class MidBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        # first layer
        nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
        # more layers
        for i in range(num_layers):
            nets.append(ResnetBlock(in_channels, in_channels, skip_scale=skip_scale))
        self.nets = nn.ModuleList(nets)
        
    def forward(self, x):
        x = self.nets[0](x)
        for net in self.nets[1:]:
            x = net(x)
        return x


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_out_channels: int,
        out_channels: int,
        num_layers: int = 1,
        upsample: bool = True,
        skip_scale: float = 1,
    ):
        super().__init__()

        nets = []
        for i in range(num_layers):
            cin = in_channels if i == 0 else out_channels
            cskip = prev_out_channels if (i == num_layers - 1) else out_channels

            nets.append(ResnetBlock(cin + cskip, out_channels, skip_scale=skip_scale))
        self.nets = nn.ModuleList(nets)

        self.upsample = None
        if upsample:
            self.upsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, xs):

        for net in self.nets:
            res_x = xs[-1]
            xs = xs[:-1]
            x = torch.cat([x, res_x], dim=1)
            x = net(x)
            
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')
            x = self.upsample(x)

        return x


# it could be asymmetric!
class RefineUnetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        # down_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        # up_channels: Tuple[int, ...] = (1024, 512, 256),
        # down_channels: Tuple[int, ...] = (64, 128, 256),
        # up_channels: Tuple[int, ...] = (256, 128, 64),
        down_channels: Tuple[int, ...] = (32, 64),
        up_channels: Tuple[int, ...] = (64, 32),
        layers_per_block: int = 2,
        skip_scale: float = np.sqrt(0.5),
        out_rescale: float = 1.,
    ):
        super().__init__()

        # first
        self.conv_in = nn.Conv2d(in_channels, down_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        down_blocks = []
        cout = down_channels[0]
        for i in range(len(down_channels)):
            cin = cout
            cout = down_channels[i]

            down_blocks.append(DownBlock(
                cin, cout, 
                num_layers=layers_per_block, 
                downsample=(i != len(down_channels) - 1), # not final layer
                skip_scale=skip_scale,
            ))
        self.down_blocks = nn.ModuleList(down_blocks)

        # mid
        self.mid_block = MidBlock(down_channels[-1], skip_scale=skip_scale)

        # up
        up_blocks = []
        cout = up_channels[0]
        for i in range(len(up_channels)):
            cin = cout
            cout = up_channels[i]
            cskip = down_channels[max(-2 - i, -len(down_channels))] # for assymetric

            up_blocks.append(UpBlock(
                cin, cskip, cout, 
                num_layers=layers_per_block + 1, # one more layer for up
                upsample=(i != len(up_channels) - 1), # not final layer
                skip_scale=skip_scale,
            ))
        self.up_blocks = nn.ModuleList(up_blocks)

        # last
        self.norm_out = nn.GroupNorm(num_channels=up_channels[-1], num_groups=32, eps=1e-5)
        self.conv_out = nn.Conv2d(up_channels[-1], out_channels, kernel_size=3, stride=1, padding=1)
        self.out_rescale = out_rescale


    def forward(self, x):
        # x: [B, Cin, H, W]

        # first
        x = self.conv_in(x)
        
        # down
        xss = [x]
        for block in self.down_blocks:
            x, xs = block(x)
            xss.extend(xs)
        
        # mid
        x = self.mid_block(x)

        # up
        for block in self.up_blocks:
            xs = xss[-len(block.nets):]
            xss = xss[:-len(block.nets)]
            x = block(x, xs)

        # last
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x) # [B, Cout, H', W']

        x = torch.tanh(self.out_rescale * x)

        return x

    