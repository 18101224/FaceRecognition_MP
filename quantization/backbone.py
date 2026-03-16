"""Minimal IR50 backbone used by the quantization workflow."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    Module,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)


class Flatten(Module):
    def forward(self, input_tensor):
        return input_tensor.view(input_tensor.size(0), -1)


class SEModule(Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=False)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    pass


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks1 = [get_block(in_channel=64, depth=64, num_units=3)]
        blocks2 = [get_block(in_channel=64, depth=128, num_units=4)]
        blocks3 = [get_block(in_channel=128, depth=256, num_units=14)]
        return blocks1, blocks2, blocks3
    raise ValueError(f"Unsupported IR depth: {num_layers}")


class Backbone(Module):
    def __init__(self, checkpoint_path: str | Path | None = None):
        super().__init__()
        blocks1, blocks2, blocks3 = get_blocks(50)

        self.input_layer = Sequential(
            Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )
        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(0.0),
            Flatten(),
            Linear(512 * 7 * 7, 512),
            BatchNorm1d(512),
        )

        def _build_body(blocks):
            modules = []
            for block in blocks:
                for bottleneck in block:
                    modules.append(
                        bottleneck_IR(
                            bottleneck.in_channel,
                            bottleneck.depth,
                            bottleneck.stride,
                        )
                    )
            return Sequential(*modules)

        self.body1 = _build_body(blocks1)
        self.body2 = _build_body(blocks2)
        self.body3 = _build_body(blocks3)

        if checkpoint_path:
            self.load_weights(checkpoint_path)

    def load_weights(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = Path(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint payload in {checkpoint_path}")
        state = {
            key[len("module."):] if key.startswith("module.") else key: value
            for key, value in state.items()
        }

        backbone_state = {
            key[len("backbone."):]: value
            for key, value in state.items()
            if key.startswith("backbone.")
        }
        if backbone_state:
            state = backbone_state

        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(
                f"Unexpected IR50 backbone weights in {checkpoint_path}: {unexpected[:5]}"
            )
        if missing:
            raise ValueError(
                f"Incomplete IR50 backbone weights in {checkpoint_path}: {missing[:5]}"
            )

    def forward(self, x, ldmk=None):
        del ldmk
        x = F.interpolate(x, size=112)
        x = self.input_layer(x)
        x1 = self.body1(x)
        x2 = self.body2(x1)
        x3 = self.body3(x2)
        x = F.adaptive_avg_pool2d(x3, 1).reshape(x.shape[0], -1)
        return x, [x1, x2, x3]
