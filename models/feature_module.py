import torch 
import torch.nn as nn 
from e2cnn import gspaces
from e2cnn import nn as enn 
from functools import partial 
from .activations import ComplexMish, Mish


__all__ = ['DeepComplexModule', 'ResidualModule']

Parameter = partial(nn.Parameter, requires_grad=True)

class ComplexLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.weight_real = Parameter(torch.randn(dim_in, dim_out))
        self.weight_imag = Parameter(torch.randn(dim_in, dim_out))
        self.bias_real = Parameter(torch.randn(dim_out))
        self.bias_imag = Parameter(torch.randn(dim_out))
    
    def forward(self, x):
        # Accept both real and complex inputs without re-wrapping complex tensors
        if torch.is_complex(x):
            real_in = x.real
            imag_in = x.imag
        else:
            real_in = x
            imag_in = torch.zeros_like(x)
        real = real_in @ self.weight_real - imag_in @ self.weight_imag + self.bias_real
        imag = real_in @ self.weight_imag + imag_in @ self.weight_real + self.bias_imag
        return torch.complex(real, imag)

class DeepComplexModule(nn.Module):
    def __init__(self, dim, depth, regular_simplex=False, num_classes=None):
        super().__init__()
        if regular_simplex : 
            blocks = [ComplexLinear(dim, num_classes-1)]
            dim = num_classes-1
        else:
            blocks = [ComplexLinear(dim, dim)]
        if depth > 1:
            blocks += [
                ComplexLinear(dim, dim) for _ in range(depth-1)
            ]
        
        self.layers = nn.Sequential(*blocks)
        self.activation = ComplexMish()
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # Start with a complex tensor only if needed
        out = x if torch.is_complex(x) else torch.complex(x, torch.zeros_like(x))
        out = self.layers[0](out)
        residual = out 
        if len(self.layers) > 1:
            for layer in self.layers[1:] : 
                out = layer(out)
                out = self.activation(out)
                out = self.norm(out.abs()) + residual.abs()
                residual = out 
        return out.abs()

class ResidualModule(nn.Module):
    def __init__(self, dim, depth, regular_simplex=False, num_classes=None):
        super().__init__()
        if regular_simplex : 
            blocks = [nn.Linear(dim, num_classes-1)]
            dim = num_classes-1
        else:
            blocks = [nn.Linear(dim, dim)]
        if depth > 1:
            blocks += [
                nn.Linear(dim, dim) for _ in range(depth-1)
            ]
        self.layers = nn.ModuleList(blocks)
        self.activation = Mish()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = x 
        residual = out
        out = self.layers[0](out)
        if len(self.layers) > 1:
            for layer in self.layers[1:] : 
                out = layer(out)    
                out = self.activation(out)
                out = self.norm(out) + residual 
                residual = out 
        return out 