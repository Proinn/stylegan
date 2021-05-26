import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.util_layers import EqualisedLinear, EqualisedConv2d, Smooth, FILTERS_PER_LAYER

class Discriminator(nn.Module):
    def __init__(self, image_size, minibatch):
        super(Discriminator, self).__init__()
        # according to the ProGAN paper (https://arxiv.org/abs/1710.10196) which uses the same discriminator blocks the amount of filters starts at 16 and increases to max 512
        size = image_size
        # color to filter layer
        self.conv_color = EqualisedConv2d(3, FILTERS_PER_LAYER[size], kernel_size=1, stride=1, padding=0)
        self.activation = nn.LeakyReLU(0.2)

        # Make just enough block layers
        self.layers = nn.ModuleList([])
        while size>4:
            self.layers.append(DiscriminatorConvBlock(FILTERS_PER_LAYER[size], FILTERS_PER_LAYER[size/2]))
            size = size/2
        self.last_block = DiscriminatorConvBlockLast(FILTERS_PER_LAYER[size], FILTERS_PER_LAYER[size], minibatch)


    def forward(self, x):
        x = self.conv_color(x)
        x = self.activation(x)
        for layer in self.layers:
            x = layer(x)
        x = self.last_block(x)
        return x
    
class DiscriminatorConvBlock(nn.Module):
    def __init__(self, filters_in, filters_out):
        super(DiscriminatorConvBlock, self).__init__()
        self.conv1 = EqualisedConv2d(filters_in, filters_in, kernel_size=3,stride=1,padding=1 )
        self.conv2 = EqualisedConv2d(filters_in, filters_out, kernel_size=3,stride=1,padding=1 ) 
        self.activation = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.smooth = Smooth()
        # Stylegan2 config E and F also have residual blocks see figure 7 (c). https://arxiv.org/abs/1912.04958
        self.residual_conv = EqualisedConv2d(filters_in, filters_out, kernel_size=1,stride=1,padding=0) 

    def forward(self, x):
        # Residual part
        residual = self.smooth(x)
        residual = self.downsample(residual)
        residual = self.residual_conv(residual)

        # Conv part
        conv = self.conv1(x)
        conv = self.activation(conv) 
        conv = self.conv2(conv)
        conv = self.activation(conv)
        conv = self.smooth(conv)
        conv = self.downsample(conv) 

        return torch.add(residual, conv) * np.sqrt(0.5)

class DiscriminatorConvBlockLast(nn.Module):
    def __init__(self, filters_in, filters_out, minibatch):
        super(DiscriminatorConvBlockLast, self).__init__()
        self.minibatch = MiniBatchStd(minibatch)
        self.conv1 = EqualisedConv2d(filters_in+1, filters_in, kernel_size=3,stride=1,padding=1 )
        self.conv2 = EqualisedConv2d(filters_in, filters_out, kernel_size=4,stride=1,padding=0 ) 
        self.activation = nn.LeakyReLU(0.2)
        self.linear = EqualisedLinear(filters_out, 1, bias=False)

    def forward(self, x):
        x = self.minibatch(x)
        x = self.conv1(x)
        x = self.activation(x) 
        x = self.conv2(x)
        x = self.activation(x) 
        x = self.linear(torch.squeeze(x))
        return  x

class MiniBatchStd(nn.Module):
    """minibatch like in ProGAN paper paragraph 3 (https://arxiv.org/abs/1710.10196)"""
    def __init__(self, minibatch):
        super(MiniBatchStd, self).__init__()
        self.minibatch = minibatch

    def forward(self, x):
        batch, channels, width, height = x.shape
        x_grouped = x.view(self.minibatch, -1, channels, width, height)
        y = torch.std(x_grouped, dim=2, keepdim=True) 
        y = torch.mean(y, dim=(3,4), keepdim=True)  
        y = torch.mean(y, dim=(0), keepdim=False)  
        y = y.repeat(self.minibatch, 1, width, height)
        x = torch.cat([x,y], dim=1)
        return  x