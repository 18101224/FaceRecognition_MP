import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from .cos_classifier import CosClassifier
from .resmodule import Residual

class ResNetClassifier(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.backbone = ResNetBackbone(freeze_bn=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.mlp1 = nn.Linear(2048,512)
        self.residual = Residual(in_features=512, num_layers=10)
        # Cosine classifier

        self.classifier = CosClassifier(in_features=512, num_classes=num_classes)
        
    def forward(self, x):
        # Backbone + pooling
        x = self.backbone(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # First residual block
        x = self.mlp1(x)
        x = self.residual(x)
        
        out = self.classifier(x)
        return out 


class ResNetBackbone(nn.Module):
    def __init__(self, pretrained=True, freeze_bn=True):
        super(ResNetBackbone, self).__init__()
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        self.norm_layer = nn.Conv2d(3,3, kernel_size=1, stride=1, padding=0, bias=True)
        # Remove the last fully connected layer and average pooling
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Freeze batch normalization layers if specified
        if freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        

    
    def forward(self, x):
        # Normalize input

        x=  self.norm_layer(x)
        # Forward through backbone
        features = self.backbone(x)
        
        return features
    
    def get_output_dim(self):
        """Get the output dimension of the backbone features."""
        return 2048  # ResNet-50 output channels
    
    def get_input_size(self):
        """Get the expected input size of the backbone."""
        return (112, 112)  # Expected input size
    
    def freeze_weights(self):
        """Freeze all weights in the backbone."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_weights(self):
        """Unfreeze all weights in the backbone."""
        for param in self.parameters():
            param.requires_grad = True 

