import torch
from torch import nn

class Residual(nn.Module):
    def __init__(self, num_classes, channels, num_layers):
        super().__init__()
        self.mlps = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.lin = nn.Linear(num_classes,channels)
        for i in range(num_layers):
            self.mlps.append(nn.Linear(channels, channels))
            self.norms.append(nn.LayerNorm(channels))
            self.norms.append(nn.LayerNorm(channels))
            self.acts.append(nn.SELU())


    def forward(self, x):
        x =
        for mlp, no norm2 , act in zip(self.mlps, self.norms[::2],self.norm[1::2], self.acts):
            r x.clone()
            xlp(x)
            x = norm1(x+res)
            x = act(x)
            x = norm2(x)
        return x


class RotationModule(nn.Module):
    def __init__(self, in_dim, num_layers, num_hidden_layers):
        super().__init__()
        self.mlps = nn.ModuleList()
        for i in range(num_layers):
            self.mlps.append(Residual(in_dim,num_hidden_layers))

    def get_relative_angle(self,weight,z):
        return torch.arccos(weight@z)

    def forward(self,x):
