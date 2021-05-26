
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np

FILTERS_PER_LAYER = {
    4: 512,
    8: 512,
    16: 256,
    32: 128,
    64: 32, 
    128: 32,
    256: 64,
    512: 32,
    1024: 16
}


class EqualisedConv2d(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size,stride,padding):
        super(EqualisedConv2d, self).__init__()
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.weights = nn.Parameter(torch.randn(self.filters_out,self.filters_in,self.kernel_size,self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(self.filters_out))
        self.weight_equaliser = np.sqrt(1/(self.filters_in*self.kernel_size*self.kernel_size))
        
    def forward(self, x):
        weights = self.weights*self.weight_equaliser
        return F.conv2d(x, weights, bias=self.bias, stride=self.stride, padding=self.padding)

class EqualisedLinear(nn.Module):
    def __init__(self, filters_in, filters_out, bias = True, bias_init=0):
        super(EqualisedLinear, self).__init__()
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.weights = nn.Parameter(torch.randn(self.filters_out,self.filters_in))
        if bias:
            self.bias = nn.Parameter(torch.ones(self.filters_out) * bias_init)
        else:
            self.bias = None
        self.weight_equaliser = np.sqrt(1/(self.filters_in))
        
    def forward(self, x):
        if self.bias is None:
            return F.linear(x, self.weights*self.weight_equaliser)
        else:
            return F.linear(x, self.weights*self.weight_equaliser, bias=self.bias)


class Smooth(nn.Module):
    """
    <a id="smooth"></a>
    ### Smoothing Layer
    This layer blurs each channel
    """

    def __init__(self):
        super().__init__()
        # Blurring kernel
        kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        # Convert the kernel to a PyTorch tensor
        kernel = torch.tensor([[kernel]], dtype=torch.float)
        # Normalize the kernel
        kernel /= kernel.sum()
        # Save kernel as a fixed parameter (no gradient updates)
        self.kernel = nn.Parameter(kernel, requires_grad=False)
        # Padding layer
        self.pad = nn.ReplicationPad2d(1)

    def forward(self, x: torch.Tensor):
        # Get shape of the input feature map
        b, c, h, w = x.shape
        # Reshape for smoothening
        x = x.view(-1, 1, h, w)

        # Add padding
        x = self.pad(x)

        # Smoothen (blur) with the kernel
        x = F.conv2d(x, self.kernel)

        # Reshape and return
        return x.view(b, c, h, w)
