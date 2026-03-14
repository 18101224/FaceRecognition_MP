"""
BatchNorm folding utilities for IR50.

IR50 block (bottleneck_IR) structure:
    res_layer: BN(in) → Conv1(3x3) → PReLU → Conv2(3x3) → BN(out)
    shortcut:  MaxPool  OR  Sequential(Conv1x1, BN)

After folding:
    res_layer: Conv1'(BN absorbed as pre-scale) → PReLU → Conv2'(BN absorbed post)
    shortcut:  MaxPool  OR  Conv1x1'
"""

import torch
import torch.nn as nn


def fold_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """Fold BatchNorm2d that immediately follows Conv2d into the Conv2d.

    Conv-BN fold:
        W' = W * (γ / √(σ² + ε))     per output channel
        b' = (b - μ) * (γ / √(σ² + ε)) + β
    """
    with torch.no_grad():
        gamma = bn.weight.data          # [out_ch]
        beta  = bn.bias.data            # [out_ch]
        mean  = bn.running_mean.data    # [out_ch]
        var   = bn.running_var.data     # [out_ch]
        std   = (var + bn.eps).sqrt()
        scale = gamma / std             # [out_ch]

        w = conv.weight.data            # [out_ch, in_ch, kH, kW]
        b = (conv.bias.data if conv.bias is not None
             else torch.zeros(conv.out_channels, device=w.device, dtype=w.dtype))

        new_w = w * scale.view(-1, 1, 1, 1)
        new_b = (b - mean) * scale + beta

    new_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        conv.kernel_size, conv.stride, conv.padding,
        conv.dilation, conv.groups,
        bias=True, padding_mode=conv.padding_mode,
    )
    new_conv = new_conv.to(device=w.device, dtype=w.dtype)
    new_conv.weight.data.copy_(new_w)
    new_conv.bias.data.copy_(new_b)
    return new_conv


def fold_bn_conv(bn: nn.BatchNorm2d, conv: nn.Conv2d) -> nn.Conv2d:
    """Fold BatchNorm2d that immediately precedes Conv2d into the Conv2d.

    BN-Conv fold:
        BN(x)[c]  = γ[c] * (x[c] − μ[c]) / σ[c] + β[c]
        Conv(BN(x)) = Σ_c W[o,c] * BN(x)[c]

        W'[o,c,kH,kW] = W[o,c,kH,kW] * γ[c] / σ[c]
        b'[o]         = b[o] + Σ_{c,kH,kW} W[o,c,kH,kW] * (β[c] − γ[c]·μ[c]/σ[c])
    """
    with torch.no_grad():
        gamma = bn.weight.data          # [in_ch]
        beta  = bn.bias.data            # [in_ch]
        mean  = bn.running_mean.data    # [in_ch]
        var   = bn.running_var.data     # [in_ch]
        std   = (var + bn.eps).sqrt()
        scale = gamma / std             # [in_ch]
        shift = beta - gamma * mean / std  # [in_ch]

        w = conv.weight.data            # [out_ch, in_ch, kH, kW]
        b = (conv.bias.data if conv.bias is not None
             else torch.zeros(conv.out_channels, device=w.device, dtype=w.dtype))

        new_w = w * scale.view(1, -1, 1, 1)
        new_b = b + (w * shift.view(1, -1, 1, 1)).sum(dim=(1, 2, 3))

    new_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels,
        conv.kernel_size, conv.stride, conv.padding,
        conv.dilation, conv.groups,
        bias=True, padding_mode=conv.padding_mode,
    )
    new_conv = new_conv.to(device=w.device, dtype=w.dtype)
    new_conv.weight.data.copy_(new_w)
    new_conv.bias.data.copy_(new_b)
    return new_conv


def _fold_bottleneck(block) -> None:
    """In-place BN fold for a single bottleneck_IR block.

    res_layer indices:
        [0] BatchNorm2d(in_channel)   ← leading BN
        [1] Conv2d(in → depth, 3x3)
        [2] PReLU(depth)
        [3] Conv2d(depth → depth, 3x3)
        [4] BatchNorm2d(depth)        ← trailing BN

    shortcut_layer:
        MaxPool2d(1, stride)          ← when in_channel == depth (no BN to fold)
        Sequential(Conv2d, BN)        ← when channels differ
    """
    res = block.res_layer
    bn_lead = res[0]
    conv1   = res[1]
    prelu   = res[2]
    conv2   = res[3]
    bn_tail = res[4]

    conv1_f = fold_bn_conv(bn_lead, conv1)   # BN → Conv: absorb into Conv1
    conv2_f = fold_conv_bn(conv2, bn_tail)   # Conv → BN: absorb into Conv2

    block.res_layer = nn.Sequential(conv1_f, prelu, conv2_f)

    # Shortcut
    sc = block.shortcut_layer
    if isinstance(sc, nn.Sequential):
        # Sequential(Conv1x1, BN)
        sc_conv, sc_bn = sc[0], sc[1]
        block.shortcut_layer = fold_conv_bn(sc_conv, sc_bn)
    # MaxPool2d shortcut needs no folding


def fold_ir50(model) -> None:
    """In-place fold all BatchNorm layers in an IR50 Backbone.

    Modifies:
      model.input_layer  : Sequential(Conv, BN, PReLU) → Sequential(Conv', PReLU)
      model.body{1,2,3}  : each bottleneck_IR block gets its BNs absorbed
    """
    # input_layer: Conv-BN-PReLU → Conv'-PReLU
    il = model.input_layer
    conv0_f = fold_conv_bn(il[0], il[1])    # Conv2d(3,64) + BN2d(64)
    model.input_layer = nn.Sequential(conv0_f, il[2])  # keep PReLU

    for body_name in ("body1", "body2", "body3"):
        body = getattr(model, body_name, None)
        if body is None:
            continue
        for block in body:
            _fold_bottleneck(block)
