# IR50 TensorRT Path

This repo now includes a deployment path for:

- FP32 IR50 checkpoint -> ONNX
- BRECQ fake-quant IR50 checkpoint -> ONNX
- BRECQ explicit Q/DQ INT8 IR50 checkpoint -> ONNX
- ONNX -> TensorRT engine via `trtexec`
- PyTorch / TensorRT accuracy evaluation on a validation set
- PyTorch / TensorRT latency benchmark

## Files

- `quantization/exporter/deploy.py`
  - inference-only wrappers for FP32, fake-quant, and Q/DQ IR50 classifiers
- `quantization/exporter/qdq.py`
  - explicit ONNX Q/DQ modules for TensorRT
- `quantization/run_tensorrt.py`
  - CLI for ONNX export, TensorRT engine build, and benchmarking
- `shells/tensorrt_ir50.sh`
  - example commands

## Prerequisites

- NVIDIA GPU runtime
- PyTorch
- `onnx`
- TensorRT `trtexec`

Optional:

- `onnxruntime-gpu` if you want a separate ONNX runtime benchmark later

## Quick Start

Export FP32:

```bash
python -m quantization.run_tensorrt export-onnx \
  --checkpoint checkpoint/<exp>/best.pth \
  --variant fp32 \
  --dynamic-batch \
  --onnx checkpoint/deploy/ir50_fp32.onnx
```

Export BRECQ fake-quant:

```bash
python -m quantization.run_tensorrt export-onnx \
  --checkpoint checkpoint/<exp>/best.pth \
  --variant brecq \
  --quant-state checkpoint/ir50_w4a4_brecq.pth \
  --onnx checkpoint/deploy/ir50_brecq_fake_quant.onnx
```

Export BRECQ explicit Q/DQ INT8:

```bash
python -m quantization.run_tensorrt export-onnx \
  --checkpoint checkpoint/<exp>/best.pth \
  --variant qdq_int8 \
  --quant-state checkpoint/ir50_w4a4_brecq.pth \
  --dynamic-batch \
  --quant-report checkpoint/deploy/ir50_brecq_qdq_int8_report.json \
  --onnx checkpoint/deploy/ir50_brecq_qdq_int8.onnx
```

Build TensorRT engine from explicit Q/DQ ONNX:

```bash
python -m quantization.run_tensorrt build-engine \
  --onnx checkpoint/deploy/ir50_brecq_qdq_int8.onnx \
  --engine checkpoint/deploy/ir50_brecq_qdq_int8.engine \
  --explicit-quantization \
  --img-size 112 \
  --min-batch 1 \
  --opt-batch 64 \
  --max-batch 128
```

Benchmark PyTorch:

```bash
python -m quantization.run_tensorrt benchmark-pytorch \
  --checkpoint checkpoint/<exp>/best.pth \
  --variant fp32 \
  --batch-size 128
```

Evaluate PyTorch accuracy:

```bash
python -m quantization.run_tensorrt eval-pytorch \
  --checkpoint checkpoint/<exp>/best.pth \
  --variant qdq_int8 \
  --quant-state checkpoint/ir50_w4a4_brecq.pth \
  --dataset-name RAF-DB \
  --dataset-path /path/to/raf-db \
  --batch-size 128
```

Benchmark TensorRT:

```bash
python -m quantization.run_tensorrt benchmark-trt \
  --engine checkpoint/deploy/ir50_fp32_fp16.engine \
  --batch-size 128 \
  --img-size 112
```

Evaluate TensorRT accuracy:

```bash
python -m quantization.run_tensorrt eval-trt \
  --checkpoint checkpoint/<exp>/best.pth \
  --engine checkpoint/deploy/ir50_brecq_qdq_int8.engine \
  --dataset-name RAF-DB \
  --dataset-path /path/to/raf-db \
  --batch-size 128
```

## Important Limitation

The `brecq` export path preserves the current repo's fake-quant graph. That is
good enough for export and deployment experiments, but it does not mean TensorRT
will execute true INT4 kernels from this graph.

The new `qdq_int8` export path emits explicit ONNX `QuantizeLinear /
DequantizeLinear` nodes so TensorRT can build a real explicit INT8 engine.
However:

- the exported ONNX tensor type is INT8, not INT4
- layers trained at 4-bit keep their narrow clip ranges inside the INT8 graph
- this preserves BRECQ's learned ranges better than the old fake-quant export,
  but it is still an INT8 TensorRT deployment path

If you pass `--quant-report <path>.json` during `export-onnx`, the exporter also
writes a per-layer report that shows:

- `input_layer` versus `body*[*]` block structure
- weight versus activation quantizers
- `trained_bits`, `qmin`, `qmax`
- per-tensor versus per-channel scale layout
- a short preview of the stored scales

If you want real low-bit acceleration beyond this explicit INT8 path, you still
need one of the following:

- ONNX Q/DQ graph that TensorRT can lower efficiently
- TensorRT plugin for the relevant quantized ops
- custom CUDA kernel path
