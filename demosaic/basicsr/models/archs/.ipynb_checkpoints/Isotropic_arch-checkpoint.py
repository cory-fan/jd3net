# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms.functional as TF

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NOTBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        #x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

# class ResBlock(nn.Module):

#     def __init__(self,channel,cnt):
#         super().__init__()
#         self.blocks = nn.ModuleList()
#         for i in range(cnt):
#             self.blocks.append(NOTBlock(channel))

#     def forward(self,x):
#         inp = x
#         for block in self.blocks:
#             x = block(x)
#         x = x+inp
#         return x

class DropPath(nn.Module):
    def __init__(self, drop_rate, module):
        super().__init__()
        self.drop_rate = drop_rate
        self.module = module

    def forward(self, feats):
        if self.training and np.random.rand() < self.drop_rate:
            return feats

        new_feats = self.module(feats)
        factor = 1. / (1 - self.drop_rate) if self.training else 1.


        if self.training and factor != 1.:
            new_feats = feats+factor*(new_feats-feats)
        return new_feats

def pad_if_necessary(im,denom):
    B,C,H,W = im.shape
    im = F.pad(im,(0,(W//denom)*denom-W,0,(H//denom)*denom-H))
    return im

class IsotropicNet(nn.Module):

    def __init__(self,img_channel=3,downsample=4,width=64,depth=16,stochastic_depth=0,start_kernel_size=3):
        super().__init__()
        self.downsample = downsample
        self.start_conv = nn.Conv2d(img_channel,start_kernel_size**2*img_channel,start_kernel_size,padding="same")
        self.down_conv = nn.Conv2d(start_kernel_size**2*img_channel,width,downsample,stride=downsample)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(DropPath(stochastic_depth,NOTBlock(width)))
        self.end_conv = nn.Sequential(nn.Conv2d(width,3*(downsample**2),1),
                                      nn.PixelShuffle(downsample))

    def forward(self,x):
        B,C,H,W = x.shape
        # # x = pad_if_necessary(x,self.downsample)
        x = self.start_conv(x)
        x = self.down_conv(x)
        y = x
        for block in self.blocks:
            x = block(x)
        x = self.end_conv(x+y)
        # x = TF.crop(x,0,0,H,W)
        return x

class MHIsotropicNet(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.network = IsotropicNet(*args,**kwargs)

    def forward(self,x):
        B = x.shape[0]
        single_bayer = x[:,0:4,:,:]
        quad_bayer = x[:,4:8,:,:]
        nona_bayer = x[:,8:12,:,:]
        x = torch.concat((single_bayer,quad_bayer,nona_bayer),dim=0)
        x = self.network(x)
        x = torch.concat((x[0:B],x[B:2*B],x[2*B:3*B]),dim=1)
        return x
        
if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]

    enc_blks = [1, 1, 1, 28]
    middle_blk_num = 1
    dec_blks = [1, 1, 1, 1]
    
    net = Isotropicx8()


    inp_shape = (3, 258, 258)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)
