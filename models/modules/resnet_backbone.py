
from torchvision.models import ResNet50_Weights, ResNeXt50_32X4D_Weights, ResNet34_Weights
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torch import Tensor 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

__all__ = ['resnet34_backbone', 'resnet50_backbone', 'resnext50_backbone', 'resnet32_backbone', 'get_resnet']

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock_s(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock_s, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def get_backbone(name, block, layers, groups=1, width_per_group=64):
    if name == 'resnet32':
        class ResNet_s(nn.Module):
            def __init__(self, block, num_blocks):
                super(ResNet_s, self).__init__()
                factor = 2
                self.in_planes = 16*factor
                self.conv1 = nn.Conv2d(3, 16 * factor, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn1 = nn.BatchNorm2d(16 * factor)
                self.layer1 = self._make_layer(block, 16 * factor, num_blocks[0], stride=1)
                self.layer2 = self._make_layer(block, 32 * factor, num_blocks[1], stride=2)
                self.layer3 = self._make_layer(block, 64 * factor, num_blocks[2], stride=2)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight)

            def _make_layer(self, block, planes, num_blocks, stride):
                strides = [stride] + [1]*(num_blocks-1)
                layers = []
                for stride in strides:
                    layers.append(block(self.in_planes, planes, stride))
                    self.in_planes = planes * block.expansion
                return nn.Sequential(*layers)

            def forward(self, x):
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = F.avg_pool2d(out, out.size()[3])
                out1 = out.view(out.size(0), -1)
                return out1
        return ResNet_s(block, layers)
    else:
        class ResNet_backbone(ResNet):
            def __init__(self, block, layers, groups=1, width_per_group=64):
                super().__init__(block=block, layers=layers, num_classes=1000, groups=groups, width_per_group=width_per_group)
            
            def _forward_impl(self, x:Tensor):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                x = torch.flatten(x, 1)

                return x
        return ResNet_backbone(block, layers, groups=groups, width_per_group=width_per_group)

def resnet34_backbone(pretrained=False):
    '''
    512 dimensions
    '''
    model = get_backbone(name='resnet34',block=BasicBlock, layers=[3,4,6,3])
    if pretrained : 
        model.load_state_dict(ResNet34_Weights.IMAGENET1K_V1.get_state_dict(progress=True))
    del model.fc 
    return model 

def resnet50_backbone(pretrained=False):
    '''
    2048 dimensions
    '''
    model = get_backbone(name='resnet50',block=Bottleneck, layers=[3,4,6,3])
    if pretrained : 
        model.load_state_dict(ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=True))
    del model.fc 
    return model 

def resnext50_backbone(pretrained=False):
    '''
    2048 dimensions
    '''
    model = get_backbone(name='resnext50',block=Bottleneck, layers=[3,4,6,3], groups=32, width_per_group=4)
    if pretrained : 
        model.load_state_dict(ResNeXt50_32X4D_Weights.IMAGENET1K_V1.get_state_dict(progress=True))
    del model.fc 
    return model 

def resnet32_backbone():
    '''
    64 dimensions
    '''
    model = get_backbone(name='resnet32',block=BasicBlock_s, layers=[5,5,5])
    return model 

def get_resnet(architecture, pretrained=False):
    if architecture == 'resnet34':
        return resnet34_backbone(pretrained=pretrained)
    elif architecture == 'resnet50':
        return resnet50_backbone(pretrained=pretrained)
    elif architecture == 'resnext50':
        return resnext50_backbone(pretrained=pretrained)
    elif architecture == 'resnet32':
        return resnet32_backbone()

