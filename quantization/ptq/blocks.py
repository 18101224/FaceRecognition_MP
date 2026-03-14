"""
Quantisable block wrappers for IR50.

Two block types are needed:
  QuantInputLayer   – wraps the Conv-PReLU input_layer  (8-bit by default)
  QuantBottleneckIR – wraps a BN-folded bottleneck_IR   (4-bit by default)

After fold_ir50() the structures are:
  input_layer   : Sequential(Conv2d, PReLU)
  bottleneck_IR : res_layer    = Sequential(Conv1, PReLU, Conv2)
                  shortcut_layer = MaxPool2d | Conv2d

Each block exposes:
  forward(x)           – quantised forward pass
  fp_forward(x)        – original full-precision forward pass (weights kept as buffers)
  opt_params()         – (adaround_v_params, lsq_scale_params)  for optimiser setup
  init_act_quantizers(x_samples)  – seed LSQ scales from representative activations
  adaround_reg(beta)   – summed AdaRound regularisation over all Convs in block
  freeze()             – commit all AdaRound decisions; switch to hard rounding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import QuantConv2d, LSQActivationQuantizer


class QuantInputLayer(nn.Module):
    """Quantised wrapper for IR50's input_layer.

    Original (after fold_ir50):
        Sequential(Conv2d(3→64, 3x3, stride=1, padding=1), PReLU(64))

    Quantised forward:
        x  →  QuantConv2d  →  PReLU  →  LSQ(act)  →  output

    The output activation is quantised so the first body block receives a
    properly quantised input.
    """

    def __init__(self, input_layer: nn.Sequential, w_bits: int = 8, a_bits: int = 8,
                 act_init_mode: str = "lsq", act_init_percentile: float = 0.999):
        super().__init__()
        conv  = input_layer[0]   # Conv2d
        prelu = input_layer[1]   # PReLU

        self.conv  = QuantConv2d(conv, w_bits=w_bits)
        self.prelu = prelu
        # PReLU output can be negative → signed LSQ
        self.act_quant = LSQActivationQuantizer(
            n_bits=a_bits,
            signed=True,
            init_mode=act_init_mode,
            init_percentile=act_init_percentile,
        )

        # Keep a reference FP copy for fp_forward
        self._fp_conv  = conv
        self._fp_prelu = prelu

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.prelu(out)
        out = self.act_quant(out)
        return out

    def fp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full-precision forward (uses the fixed-buffer weights directly)."""
        out = F.conv2d(x, self.conv.weight, self.conv.bias,
                       self.conv.stride, self.conv.padding,
                       self.conv.dilation, self.conv.groups)
        out = self.prelu(out)
        return out

    # ------------------------------------------------------------------
    def init_act_quantizers(self, x_samples: torch.Tensor) -> None:
        """Seed LSQ scale from a representative batch of inputs."""
        with torch.no_grad():
            out = F.conv2d(x_samples, self.conv.weight, self.conv.bias,
                           self.conv.stride, self.conv.padding,
                           self.conv.dilation, self.conv.groups)
            out = self.prelu(out)
        self.act_quant.init_from_data(out)

    def opt_params(self):
        """Return (adaround_v_list, lsq_scale_list)."""
        ada = [self.conv.weight_quantizer.v]
        lsq = [self.act_quant.scale]
        return ada, lsq

    def adaround_reg(self, beta: float) -> torch.Tensor:
        return self.conv.weight_quantizer.regularization(beta)

    def freeze(self) -> None:
        self.conv.weight_quantizer.freeze()


# ---------------------------------------------------------------------------

class QuantBottleneckIR(nn.Module):
    """Quantised wrapper for a BN-folded bottleneck_IR block.

    Expected input structure (after fold_ir50):
        res_layer       : Sequential(Conv1, PReLU, Conv2)
        shortcut_layer  : MaxPool2d  OR  Conv2d  (BN already folded in)

    Quantised forward:
        x  ──┬──▶  Conv1(quant_w)  →  PReLU  →  LSQ(act)  →  Conv2(quant_w)  ──┐
             │                                                                    ▼
             └──▶  shortcut(quant_w or pool)  ─────────────────────────────────  + ▶ output

    The residual addition output is the reconstruction target for BRECQ.
    It is NOT additionally quantised here; the next block's Conv1 (which
    absorbs the leading BN) implicitly scales the distribution.
    """

    def __init__(self, block, w_bits: int = 4, a_bits: int = 4,
                 act_init_mode: str = "lsq", act_init_percentile: float = 0.999):
        super().__init__()
        res = block.res_layer          # Sequential(Conv1, PReLU, Conv2)
        sc  = block.shortcut_layer     # MaxPool2d or Conv2d

        self.conv1 = QuantConv2d(res[0], w_bits=w_bits)
        self.prelu = res[1]            # PReLU kept in FP32 (no weights to quantise)
        # PReLU output → signed activation quantiser
        self.act_quant = LSQActivationQuantizer(
            n_bits=a_bits,
            signed=True,
            init_mode=act_init_mode,
            init_percentile=act_init_percentile,
        )
        self.conv2 = QuantConv2d(res[2], w_bits=w_bits)

        if isinstance(sc, nn.MaxPool2d):
            self.shortcut      = sc
            self._shortcut_quant = None     # pooling has no weights
        else:
            self.shortcut        = QuantConv2d(sc, w_bits=w_bits)
            self._shortcut_quant = self.shortcut

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.shortcut(x)

        r = self.conv1(x)
        r = self.prelu(r)
        r = self.act_quant(r)
        r = self.conv2(r)

        return r + s

    def fp_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full-precision forward using the fixed buffer weights."""
        # shortcut
        if isinstance(self.shortcut, nn.MaxPool2d):
            s = self.shortcut(x)
        else:
            s = F.conv2d(x, self.shortcut.weight, self.shortcut.bias,
                         self.shortcut.stride, self.shortcut.padding,
                         self.shortcut.dilation, self.shortcut.groups)

        r = F.conv2d(x, self.conv1.weight, self.conv1.bias,
                     self.conv1.stride, self.conv1.padding,
                     self.conv1.dilation, self.conv1.groups)
        r = self.prelu(r)
        r = F.conv2d(r, self.conv2.weight, self.conv2.bias,
                     self.conv2.stride, self.conv2.padding,
                     self.conv2.dilation, self.conv2.groups)
        return r + s

    # ------------------------------------------------------------------
    def init_act_quantizers(self, x_samples: torch.Tensor) -> None:
        """Seed LSQ scale from representative inputs (no gradient)."""
        with torch.no_grad():
            r = F.conv2d(x_samples, self.conv1.weight, self.conv1.bias,
                         self.conv1.stride, self.conv1.padding,
                         self.conv1.dilation, self.conv1.groups)
            r = self.prelu(r)
        self.act_quant.init_from_data(r)

    def opt_params(self):
        """Return (adaround_v_list, lsq_scale_list) for optimiser construction."""
        ada = [self.conv1.weight_quantizer.v, self.conv2.weight_quantizer.v]
        if self._shortcut_quant is not None:
            ada.append(self._shortcut_quant.weight_quantizer.v)
        lsq = [self.act_quant.scale]
        return ada, lsq

    def adaround_reg(self, beta: float) -> torch.Tensor:
        reg = (self.conv1.weight_quantizer.regularization(beta)
               + self.conv2.weight_quantizer.regularization(beta))
        if self._shortcut_quant is not None:
            reg = reg + self._shortcut_quant.weight_quantizer.regularization(beta)
        return reg

    def freeze(self) -> None:
        self.conv1.weight_quantizer.freeze()
        self.conv2.weight_quantizer.freeze()
        if self._shortcut_quant is not None:
            self._shortcut_quant.weight_quantizer.freeze()


# ---------------------------------------------------------------------------
# Helper: build the ordered list of quant blocks from a (folded) IR50 model
# ---------------------------------------------------------------------------

def build_quant_blocks(folded_model, w_bits: int = 4, a_bits: int = 4,
                       first_layer_bits: int = 8,
                       act_init_mode: str = "lsq",
                       act_init_percentile: float = 0.999):
    """Return an ordered list of quant blocks for a BN-folded IR50 Backbone.

    Block order matches the forward pass:
        [0]   QuantInputLayer      (first_layer_bits)
        [1–3] QuantBottleneckIR    body1[0..2]
        [4–7] QuantBottleneckIR    body2[0..3]
        [8–21] QuantBottleneckIR   body3[0..13]

    Returns:
        blocks : list of QuantInputLayer | QuantBottleneckIR
        names  : list of str (block identifiers)
    """
    blocks = []
    names  = []

    blocks.append(QuantInputLayer(folded_model.input_layer,
                                  w_bits=first_layer_bits,
                                  a_bits=first_layer_bits,
                                  act_init_mode=act_init_mode,
                                  act_init_percentile=act_init_percentile))
    names.append("input_layer")

    for body_name in ("body1", "body2", "body3"):
        body = getattr(folded_model, body_name, None)
        if body is None:
            continue
        for i, blk in enumerate(body):
            blocks.append(QuantBottleneckIR(
                blk,
                w_bits=w_bits,
                a_bits=a_bits,
                act_init_mode=act_init_mode,
                act_init_percentile=act_init_percentile,
            ))
            names.append(f"{body_name}[{i}]")

    return blocks, names
