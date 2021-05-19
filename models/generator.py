import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

FILTERS_PER_LAYER = {
    4: 512,
    8: 512,
    16: 512,
    32: 512,
    64: 256, 
    128: 128,
    256: 64,
    512: 32,
    1024: 16
}

class Generator(nn.Module):
    def __init__(self, image_size, n_mapping_layers=8, z_dim=512, w_dim=512, mapping_hidden_dim=512, stylemixing=0):
        super(Generator, self).__init__()
        self.stylemixing = stylemixing
        self.mapping_network = MappingNetwork(n_mapping_layers, z_dim, w_dim, mapping_hidden_dim)
        self.layers_n = int(math.log(image_size,2)-1)
        self.generator_module = GeneratorConvStack(self.layers_n, w_dim)

        

    def forward(self, x):
        styles = self.mapping_network(x)
        if self.stylemixing>random.uniform(0,1):
            # duplicate styles for amount of layers, and mix them
            # TODO: implement stylemixing by shuffeling the repeated styles 
            pass
        else: 
            # duplicate styles for amount of layers
            batch, w_dim = styles.shape
            styles = styles.view(batch, w_dim, 1)
            styles = styles.repeat(1,1,self.layers_n)
        
        images = self.generator_module(styles)
        images = torch.tanh(images)
        return images


class GeneratorConvStack(nn.Module):
    def __init__(self, layers_n, w_dim):
        super(GeneratorConvStack, self).__init__()
        self.gen_layers = nn.ModuleList([])
        size = 2
        for i in range(int(layers_n)):
            if i == 0:
                self.gen_layers.append(FirstGenBlock(FILTERS_PER_LAYER[size*2], w_dim))
            else:
                self.gen_layers.append(GenBlockBlock(FILTERS_PER_LAYER[size], FILTERS_PER_LAYER[size*2], w_dim))
            size = size * 2
    
    def forward(self, styles):
        for i, layer in enumerate(self.gen_layers):
            if i == 0:
                x, rgb = layer(styles[:,:,i])
            else: 
                x, rgb_residual = layer(x, styles[:,:,i])
                rgb = F.interpolate(rgb, mode='bilinear',scale_factor=2)
                rgb = torch.add(rgb, rgb_residual)
        return rgb

class MappingNetwork(nn.Module):
    def __init__(self, n_mapping_layers, z_dim, w_dim,mapping_hidden_dim):
        super(MappingNetwork, self).__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_mapping_layers):
            input_dim = z_dim if i == 0 else mapping_hidden_dim
            output_dim = w_dim if i == (n_mapping_layers-1) else mapping_hidden_dim
            self.layers.append(nn.Linear(input_dim, output_dim))
        self.activation = nn.LeakyReLU(0.2)
        self.pixnorm = PixNorm()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.pixnorm(x)
        return x


class ToRGB(nn.Module):
    def __init__(self, channels):
        super(ToRGB, self).__init__()
        self.conv_layer = nn.Conv2d(channels, 3,1)

    def forward(self, x):
        x = self.conv_layer(x)
        return x

class FirstGenBlock(nn.Module):
    def __init__(self, channels, w_dim, init_img_size=4):
        super(FirstGenBlock, self).__init__()
        self.weights = nn.Parameter(torch.randn(1, channels, init_img_size, init_img_size))
        self.noise1 = NoiseAddition(channels)
        self.conv2 = ModulationConv(channels, channels, w_dim=w_dim)
        self.noise2 = NoiseAddition(channels)
        self.activation = nn.ReLU()
        self.to_rgb = ToRGB(channels)
    
    def forward(self, styles):
        batch, z_dim = styles.shape
        x = self.weights.repeat(batch, 1, 1, 1)
        x = self.noise1(x)
        x = self.conv2(x, styles)
        x = self.activation(x)
        x = self.noise2(x)
        rgb = self.to_rgb(x)
        return x, rgb


class GenBlockBlock(nn.Module):
    def __init__(self, channels_in, channels_out, w_dim):
        super(GenBlockBlock, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = ModulationConv(channels_in, channels_out, w_dim=w_dim)
        self.noise1 = NoiseAddition(channels_out)
        self.conv2 = ModulationConv(channels_out, channels_out, w_dim=w_dim)
        self.noise2 = NoiseAddition(channels_out)
        self.activation = nn.LeakyReLU(0.2)
        self.to_rgb = ToRGB(channels_out)

    
    def forward(self, x, styles):
        x = self.upsample(x)
        x = self.conv1(x, styles)
        x = self.activation(x)
        x = self.noise1(x)
        x = self.conv2(x, styles)
        x = self.activation(x)
        x = self.noise2(x)
        rgb = self.to_rgb(x)
        return x, rgb



class PixNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        y = x.pow(2.).mean(dim=1, keepdim=True).add(self.epsilon).sqrt() 
        return x / y 

class ModulationConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, w_dim=512, epsilon=1e-8):
        super(ModulationConv, self).__init__()
        self.out_channel = out_channel
        self.in_channel = in_channel
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros([1,out_channel,1,1]))
        
        self.epsilon = epsilon
        self.affline = nn.Linear(w_dim, in_channel, bias=False)


    def forward(self, x, styles):
        batch, in_channel, height, width = x.shape

        z = self.affline(styles)
        z = z.view(batch, 1, in_channel, 1, 1)
        weights = self.weights * z
        demodulate = torch.rsqrt(weights.pow(2).sum([2, 3, 4]) + 1e-8)
        weights = weights * demodulate.view(batch, self.out_channel, 1, 1, 1)

        # reshape to groups
        weights = weights.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        x = x.view(1, batch * in_channel, height, width)
        # do convolution
        out = F.conv2d(x,weights, bias=None, stride=1, padding=1, groups=batch)

        # reshape back to batches
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)
        # add biases
        out = torch.add(out, self.bias)

        return out

class NoiseAddition(nn.Module):
    def __init__(self, channels):
        super(NoiseAddition, self).__init__()
        
        self.scaling_factor = nn.Parameter(torch.randn([1,channels,1,1]))

    def forward(self, x):
        batch, _ , height, width = x.shape

        # scaled gaussian noise
        noise = torch.randn([batch,1,height,width])
        scaled_noise = torch.multiply(noise, self.scaling_factor)
        x = torch.add(x, scaled_noise)
        return x


if __name__=='__main__':
    generator = Generator(128)

    input = torch.randn(4,512)
    print(generator(input))
