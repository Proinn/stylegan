import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from models.util_layers import EqualisedLinear, Smooth, FILTERS_PER_LAYER

class Generator(nn.Module):
    def __init__(self, image_size, n_mapping_layers=8, z_dim=512, w_dim=512, mapping_hidden_dim=512, stylemixing=0.9):
        super(Generator, self).__init__()
        self.stylemixing = stylemixing
        self.mapping_network = MappingNetwork(n_mapping_layers, z_dim, w_dim, mapping_hidden_dim)
        self.layers_n = int(math.log(image_size,2)-1)
        self.synthesis_network = SynthesisNetwork(self.layers_n, w_dim)
        

    def forward(self, x):
        
        if self.stylemixing>random.uniform(0,1):
            # duplicate styles for amount of layers, and mix them
            styles_1 = self.mapping_network(x)
            styles_2 = self.mapping_network(x)
            batch, w_dim = styles_1.shape
            styles_1 = styles_1.view(batch, w_dim, 1)
            styles_2 = styles_2.view(batch, w_dim, 1)
            split = random.randrange(1, self.layers_n-1)
            styles_1 = styles_1.repeat(1,1,split)
            styles_2 = styles_2.repeat(1,1,self.layers_n-split)
            styles = torch.cat([styles_1,styles_2], dim=2)
        else: 
            # duplicate styles for amount of layers
            styles = self.mapping_network(x)
            batch, w_dim = styles.shape
            styles = styles.view(batch, w_dim, 1)
            styles = styles.repeat(1,1,self.layers_n)
        
        images = self.synthesis_network(styles)
        #images = torch.tanh(images)
        return images


class SynthesisNetwork(nn.Module):
    def __init__(self, layers_n, w_dim):
        super(SynthesisNetwork, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.gen_layers = nn.ModuleList([])
        self.smooth = Smooth()
        size = 2
        for i in range(int(layers_n)):
            if i == 0:
                self.gen_layers.append(FirstGenBlock(FILTERS_PER_LAYER[size*2], w_dim))
            else:
                self.gen_layers.append(GenBlock(FILTERS_PER_LAYER[size], FILTERS_PER_LAYER[size*2], w_dim))
            size = size * 2
    
    def forward(self, styles):
        for i, layer in enumerate(self.gen_layers):
            if i == 0:
                x, rgb = layer(styles[:,:,i])
            else: 
                x, rgb_residual = layer(x, styles[:,:,i])
                rgb = self.upsample(rgb)
                rgb = self.smooth(rgb)
                rgb = torch.add(rgb, rgb_residual)
        return rgb

class MappingNetwork(nn.Module):
    def __init__(self, n_mapping_layers, z_dim, w_dim,mapping_hidden_dim):
        super(MappingNetwork, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_mapping_layers):
            input_dim = z_dim if i == 0 else mapping_hidden_dim
            output_dim = w_dim if i == (n_mapping_layers-1) else mapping_hidden_dim
            self.layers.append(EqualisedLinear(input_dim, output_dim, bias=True, bias_init=0))
        self.activation = nn.LeakyReLU(0.2)
        self.pixnorm = PixNorm()
        self.activation_gain = np.sqrt(2)


    def forward(self, x):
        x = self.pixnorm(x)
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x) * self.activation_gain

        return x
    

class ToRGB(nn.Module):
    def __init__(self, channels):
        super(ToRGB, self).__init__()
        self.conv_layer = ModulationConv(in_channel = channels, out_channel=3,kernel_size = 1, stride = 1, padding = 0, demodulate = True)

    def forward(self, x, styles):
        x = self.conv_layer(x, styles)
        return x
    
class FirstGenBlock(nn.Module):
    def __init__(self, channels, w_dim, init_img_size=4):
        super(FirstGenBlock, self).__init__()
        self.weights = nn.Parameter(torch.ones(1, channels, init_img_size, init_img_size))
        self.noise1 = NoiseAddition(channels)
        self.conv2 = ModulationConv(channels, channels, w_dim=w_dim)
        self.noise2 = NoiseAddition(channels)
        self.activation = nn.LeakyReLU(0.2)
        self.activation_gain = np.sqrt(2)
        self.to_rgb = ToRGB(channels)
        
    
    def forward(self, styles):
        batch, z_dim = styles.shape
        x = self.weights.repeat(batch, 1, 1, 1)
        x = self.noise1(x)
        x = self.conv2(x, styles)
        x = self.activation(x) * self.activation_gain
        x = self.noise2(x)
        rgb = self.to_rgb(x, styles)
        return x, rgb


class GenBlock(nn.Module):
    def __init__(self, channels_in, channels_out, w_dim):
        super(GenBlock, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = ModulationConv(channels_in, channels_out, w_dim=w_dim)
        self.noise1 = NoiseAddition(channels_out)
        self.conv2 = ModulationConv(channels_out, channels_out, w_dim=w_dim)
        self.noise2 = NoiseAddition(channels_out)
        self.activation = nn.LeakyReLU(0.2)
        self.activation_gain = np.sqrt(2)
        self.to_rgb = ToRGB(channels_out)
        self.smooth = Smooth()

    
    def forward(self, x, styles):
        x = self.upsample(x)
        x = self.smooth(x)
        x = self.conv1(x, styles)
        x = self.activation(x) * self.activation_gain
        x = self.noise1(x)
        x = self.conv2(x, styles)
        x = self.activation(x) * self.activation_gain
        x = self.noise2(x)
        rgb = self.to_rgb(x, styles)
        return x, rgb


class PixNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(self.epsilon).rsqrt() 
        return x * y 

class ModulationConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3,padding = 1, stride = 1,w_dim=512, epsilon=1e-8, demodulate= True):
        super(ModulationConv, self).__init__()
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros([1,out_channel,1,1]))
        self.weights_ElR = np.sqrt(1/(in_channel * kernel_size * kernel_size))
        self.epsilon = epsilon
        self.affline = EqualisedLinear(w_dim, in_channel, bias=True, bias_init=1)
        self.padding = padding
        self.stride = stride
        self.demodulate = demodulate


    def forward(self, x, styles):
        batch, in_channel, height, width = x.shape
        
        z = self.affline(styles)
        z = z.view(batch, 1, in_channel, 1, 1)
        
        weights = self.weights * z * self.weights_ElR
        
        if self.demodulate:
            demodulate = torch.rsqrt(weights.pow(2).sum(dim = [2, 3, 4], keepdim = True) + 1e-8)
            weights = weights * demodulate

        # reshape to groups
        weights = weights.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        
        x = x.view(1, batch * in_channel, height, width)
        # do convolution
        out = F.conv2d(x, weights, padding=self.padding, stride = self.stride, groups=batch)

        # reshape back to batches
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)
        # add biases
        out = torch.add(out, self.bias)

        return out


class NoiseAddition(nn.Module):
    def __init__(self, channels):
        super(NoiseAddition, self).__init__()
        
        self.scaling_factor = nn.Parameter(torch.zeros([1,channels,1,1]))

    def forward(self, x):
        batch, _ , height, width = x.shape
        device = x.device
        # scaled gaussian noise
        noise = torch.randn([batch,1,height,width], device = device)
        scaled_noise = torch.multiply(noise, self.scaling_factor)
        x = torch.add(x, scaled_noise)
        return x


if __name__=='__main__':
    generator = Generator(16)

    input = torch.randn(4,512)
    print(generator(input))