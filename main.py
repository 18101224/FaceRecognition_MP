from models import *
import torch
from torch import nn

def get_ir():
    back = Backbone(50,0.0,'ir')
    back = load_pretrained_weights(back,torch.load('checkpoint/ir50.pth'))
    class IR(nn.Module):
        def __init__(self,backbone):
            super().__init__()
            self.backbone = backbone
            self.classifier = nn.Sequential(
                nn.AvgPool2d(kernel_size=14),
                nn.BatchNorm1d(256),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(7)
            )
        def forward(self,x):
            _,_,x = self.backbone(x)
            return x, self.classifier(x)
    return IR(back)



