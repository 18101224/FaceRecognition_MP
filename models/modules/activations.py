import torch 
import torch.nn as nn 

__all__ = ['ComplexMish', 'Mish']

class ComplexMish(nn.Module):
    def forward(self, x):
        real = x.real * torch.tanh(nn.functional.softplus(x.real))
        imag = x.imag * torch.tanh(nn.functional.softplus(x.imag))
        return torch.complex(real, imag)
    

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(nn.functional.softplus(x))