import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Discriminator(nn.Module):
    def __init__(self, image_size, minibatch, min_filters=265, max_filters=512):
        super(Discriminator, self).__init__()
        # according to the ProGAN paper (https://arxiv.org/abs/1710.10196) which uses the same discriminator blocks the amount of filters starts at 16 and increases to max 512
        filters = min_filters
        # color to filter layer
        self.conv_color = nn.Conv2d(3, filters, kernel_size=1, stride=1, padding=0)
        self.activation = nn.LeakyReLU(0.2)

        # Make just enough block layers
        layers_n = math.log(image_size,2)-2
        self.layers = nn.ModuleList([])
        for i in range(int(layers_n)):
            filters_in = filters
            filters_out = min(max_filters, filters*2)
            self.layers.append(DiscriminatorConvBlock(filters_in, filters_out))
            filters = filters_out
        self.last_block = DiscriminatorConvBlockLast(filters, filters, minibatch)


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
        self.conv1 = nn.Conv2d(filters_in, filters_in, kernel_size=3,stride=1,padding=1 )
        self.conv2 = nn.Conv2d(filters_in, filters_out, kernel_size=3,stride=1,padding=1 ) 
        self.activation = nn.LeakyReLU(0.2)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        # Stylegan2 config E and F also have residual blocks see figure 7 (c). https://arxiv.org/abs/1912.04958
        self.residual_conv = nn.Conv2d(filters_in, filters_out, kernel_size=1,stride=1,padding=0) 

    def forward(self, x):
        # Residual part
        residual = self.downsample(x)
        residual = self.residual_conv(residual)

        # Conv part
        conv = self.conv1(x)
        conv = self.activation(conv)
        conv = self.conv2(conv)
        conv = self.activation(conv)
        conv = self.downsample(conv)

        return torch.add(residual, conv) 

class DiscriminatorConvBlockLast(nn.Module):
    def __init__(self, filters_in, filters_out, minibatch):
        super(DiscriminatorConvBlockLast, self).__init__()
        self.minibatch = MiniBatchStd(minibatch)
        self.conv1 = nn.Conv2d(filters_in+1, filters_in, kernel_size=3,stride=1,padding=1 )
        self.conv2 = nn.Conv2d(filters_in, filters_out, kernel_size=4,stride=1,padding=0 ) 
        self.activation = nn.LeakyReLU(0.2)
        self.linear = nn.Linear(filters_out, 1, bias=True)

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

if __name__=='__main__':
    d = Discriminator(1024, 4)
    i = torch.ones((8,3,1024,1024))
    print(d(i).size())