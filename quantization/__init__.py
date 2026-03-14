"""Facade for the quantization package.

Package layout:
  - ``quantization.calibration``: BN folding and FP32 IR50 wrappers
  - ``quantization.ptq``: fake-quant modules and BRECQ optimizer
  - ``quantization.exporter``: ONNX/TensorRT export helpers
"""

from . import calibration, exporter, ptq
from .calibration import fold_ir50, fold_conv_bn, fold_bn_conv, IR50FPModel
from .ptq import (
    round_ste,
    AdaRoundQuantizer,
    LSQActivationQuantizer,
    QuantConv2d,
    QuantLinear,
    QuantBottleneckIR,
    QuantInputLayer,
    build_quant_blocks,
    BRECQOptimizer,
)
from .exporter import (
    QuantizedIR50Classifier,
    QDQIR50Classifier,
    export_ir50_onnx,
    load_brecq_ir50_classifier,
    load_brecq_qdq_ir50_classifier,
    load_brecq_qdq_ir50_export_bundle,
    load_fp32_ir50_classifier,
    load_ir50_model_params,
    build_qdq_export_report,
    save_qdq_export_report,
)

__all__ = [
    "calibration",
    "ptq",
    "exporter",
    "fold_ir50",
    "fold_conv_bn",
    "fold_bn_conv",
    "IR50FPModel",
    "round_ste",
    "AdaRoundQuantizer",
    "LSQActivationQuantizer",
    "QuantConv2d",
    "QuantLinear",
    "QuantBottleneckIR",
    "QuantInputLayer",
    "build_quant_blocks",
    "BRECQOptimizer",
    "QuantizedIR50Classifier",
    "QDQIR50Classifier",
    "export_ir50_onnx",
    "load_brecq_ir50_classifier",
    "load_brecq_qdq_ir50_classifier",
    "load_brecq_qdq_ir50_export_bundle",
    "load_fp32_ir50_classifier",
    "load_ir50_model_params",
    "build_qdq_export_report",
    "save_qdq_export_report",
]
