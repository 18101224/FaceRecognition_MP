import torch
import torch.nn as nn
import torch.nn.functional as F

import e2cnn
from e2cnn import gspaces
from e2cnn.nn import FieldType
from e2cnn.nn import GeometricTensor
from e2cnn.nn.modules.r2_conv import R2Conv
from e2cnn.nn.modules.equivariant_module import EquivariantModule
from e2cnn.nn.modules.batchnormalization.inner import InnerBatchNorm  # v0.2.3 호환

class E2BasicBlock(EquivariantModule):
    def __init__(self, in_type: FieldType, out_type: FieldType, stride: int = 1):
        super().__init__()
        self.in_type = in_type
        self.out_type = out_type
        
        # Conv1
        self.conv1 = R2Conv(in_type, out_type, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = InnerBatchNorm(out_type)  # Equivariant BatchNorm
        # Conv2
        self.conv2 = R2Conv(out_type, out_type, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = InnerBatchNorm(out_type)
        
        # Shortcut (projection if needed)
        self.shortcut = None
        if in_type.size != out_type.size or stride != 1:
            self.shortcut = R2Conv(in_type, out_type, kernel_size=1, padding=0, stride=stride, bias=False)

    def forward(self, x: GeometricTensor) -> GeometricTensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = GeometricTensor(F.relu(out.tensor), self.out_type)  # ReLU 후 re-wrap (v0.2.3 스타일)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Shortcut
        shortcut = x if self.shortcut is None else self.shortcut(x)
        out = GeometricTensor(out.tensor + shortcut.tensor, self.out_type)
        
        out = GeometricTensor(F.relu(out.tensor), self.out_type)
        return out

    def evaluate_output_shape(self, input_shape):
        return input_shape  # stride=1로 shape 유지

class E2ResNet32(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Group space: Rotations by 90 degrees (N=4, group order=4) - 대문자 R로 수정
        self.r2_act = gspaces.Rot2dOnR2(N=4)
        
        # Input type: 3 trivial representations (RGB channels)
        self.in_type = FieldType(self.r2_act, 3 * [self.r2_act.trivial_repr])
        
        # Initial conv: to 16 regular reps (dim=16*4=64)
        feat_type = FieldType(self.r2_act, 16 * [self.r2_act.regular_repr])
        self.conv1 = R2Conv(self.in_type, feat_type, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = InnerBatchNorm(feat_type)
        self.in_type = feat_type  # Update in_type
        
        # Layers: 3 groups of 5 BasicBlocks each, channels 16→32→64 (all regular)
        self.layer1 = self._make_layer(16, 5, stride=1)  # 16 regular (dim=64)
        self.layer2 = self._make_layer(32, 5, stride=1)  # 32 regular (dim=128)
        self.layer3 = self._make_layer(64, 5, stride=1)  # 64 regular (dim=256)
        
        # Final linear: from 256 dim to num_classes
        self.linear = nn.Linear(256, num_classes)  # 64 multiplicity * 4 = 256

    def _make_layer(self, multiplicity: int, num_blocks: int, stride: int):
        out_type = FieldType(self.r2_act, multiplicity * [self.r2_act.regular_repr])
        layers = [E2BasicBlock(self.in_type, out_type, stride=stride)]
        self.in_type = out_type  # Update for next layer
        for _ in range(1, num_blocks):
            layers.append(E2BasicBlock(out_type, out_type, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        # Wrap input as GeometricTensor
        x = GeometricTensor(x, self.in_type)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = GeometricTensor(F.relu(out.tensor), self.in_type)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Adaptive pooling to handle any resolution (e.g., 32x32 → 1x1)
        out_tensor = out.tensor  # (B, 256, H, W)  # dim=256 (64*4)
        out_tensor = F.adaptive_avg_pool2d(out_tensor, (1, 1))  # (B, 256, 1, 1)
        out_tensor = out_tensor.view(out_tensor.size(0), -1)  # (B, 256)
        
        out = self.linear(out_tensor)
        return out

# 모델 인스턴스화
def e2_resnet32(num_classes=10):
    return E2ResNet32(num_classes)

# 테스트 함수
def test():
    model = e2_resnet32()
    # 32x32 입력 테스트
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)  # torch.Size([1, 10])

# test()
