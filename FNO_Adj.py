"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 1D problem Training a FNO on RCWA Adjoint Gradient 
[paper](https://arxiv.org/pdf/2010.08895.pdf)
=============================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import operator
from functools import reduce
from functools import partial

#import sys
#sys.path.append('C:\\Users\\user\\neuraloperator\\master')
from utilities3 import *

torch.manual_seed(0)
np.random.seed(0)

#Complex multiplication
def compl_mul1d(a, b):
    # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
    op = partial(torch.einsum, "bix,iox->box")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),

        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        #x_ft = torch.rfft(x, 1, normalized=True, onesided=True)
        #x_comp = torch.view_as_complex(x)
        x_ft = torch.fft.rfft(x,dim =-1,norm = 'ortho')
        x_ft = torch.view_as_real(x_ft)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1] = compl_mul1d(x_ft[:, :, :self.modes1], self.weights1) ## Weight FFT of input, truncate at max F-mode, which is modes1.Train weights1 as it is a Parameter 

        #Return to physical space
        #x = torch.ifft(out_ft, 1, normalized=True, onesided=True, signal_sizes=(x.size(-1), ))
        out_ft = torch.view_as_complex(out_ft)
        x = torch.fft.irfft(out_ft, dim=-1, norm = 'ortho')
        return x

class SimpleBlock1d(nn.Module):
    def __init__(self, modes, width):
        super(SimpleBlock1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        '''
        '''
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(1, self.width) # input channel is 2: (a(x), x), output channel is width, which will go through layers. a in this case is pattern (1, -1)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        """
        All ftns from torch.nn
        """
        
    def forward(self, x):

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x) # Process of FT -> R -> IFT
        x2 = self.w1(x) # Linear Transformation (Non-periodic term)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class FNO1D(nn.Module):
    def __init__(self, modes, width):
        super(FNO1D, self).__init__()
        """
        A wrapper function
        """
        self.conv1 = SimpleBlock1d(modes, width)
    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()
    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))
        return c