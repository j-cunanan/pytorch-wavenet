import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

def constant_pad_1d(input, target_size, dimension=0, value=0, pad_start=False):
    """Modern implementation of constant padding using F.pad"""
    num_pad = target_size - input.size(dimension)
    assert num_pad >= 0, 'target size has to be greater than input size'
    
    if num_pad == 0:
        return input

    # Convert padding to the format expected by F.pad
    if dimension == 0:
        pad_dims = (0, 0, 0, 0, num_pad if pad_start else 0, 0 if pad_start else num_pad)
    elif dimension == 1:
        pad_dims = (0, 0, num_pad if pad_start else 0, 0 if pad_start else num_pad, 0, 0)
    else:  # dimension == 2
        pad_dims = (num_pad if pad_start else 0, 0 if pad_start else num_pad, 0, 0, 0, 0)

    return F.pad(input, pad_dims, mode='constant', value=value)

def dilate(x, dilation, init_dilation=1, pad_start=True):
    """
    :param x: Tensor of size (N, C, L)
    :param dilation: Target dilation
    :param init_dilation: Initial dilation
    :param pad_start: Whether to pad at start or end
    :return: Dilated tensor
    """
    [n, c, l] = x.size()
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    l_old = int(round(l / dilation_factor))
    n_old = int(round(n * dilation_factor))
    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x

class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, dtype=torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.dtype = dtype
        if data is None:
            self.data = torch.zeros(num_channels, max_length)
        else:
            self.data = data

    def enqueue(self, input):
        # Handle input dimensions properly
        if input.dim() == 2:
            input = input.squeeze(1)  # Remove the second dimension if it exists
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]
        
        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def to(self, device):
        self.data = self.data.to(device)
        return self

    def reset(self):
        self.data.zero_()
        self.in_pos = 0
        self.out_pos = 0