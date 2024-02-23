import torch
import torch.nn as nn


# from https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
class Pix2PixDecoder(nn.Module):
    def __init__(self, input_nc=256, output_nc=3, ngf=256, n_upsampling=0, n_blocks=6, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect', out_rescale=1):
        assert(n_blocks >= 0)
        super(Pix2PixDecoder, self).__init__()        
        self.out_rescale = out_rescale

        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        ### resnet blocks
        for i in range(n_blocks):
            model += [ResnetBlock(ngf, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample      
        mult = 2 # needed when n_upsampling=0
        for i in range(n_upsampling):
            mult = 2**(-i)
            model += [nn.ConvTranspose2d(int(ngf * mult), int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        # import pdb; pdb.set_trace()
        model += [nn.ReflectionPad2d(3), nn.Conv2d(int(ngf * mult / 2), output_nc, kernel_size=7, padding=0)]        
        self.model = nn.Sequential(*model)

        # self.models = model
            
    def forward(self, input):
        # for i,m in enumerate(self.models):
        #     print(i)
        #     input = m.to(input.device)(input)
        out = self.model(input)        
        out = torch.tanh(out * self.out_rescale)
        return out
    
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
