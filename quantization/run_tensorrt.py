#!/usr/bin/env python3
"""
TensorRT deployment helpers for IR50.

Typical flow:
  1. Export PyTorch model to ONNX
  2. Build a TensorRT engine with trtexec
  3. Benchmark PyTorch and TensorRT latency separately

Examples:
  python -m quantization.run_tensorrt export-onnx \
      --checkpoint checkpoint/exp/best.pth \
      --variant fp32 \
      --onnx out/ir50_fp32.onnx

  python -m quantization.run_tensorrt export-onnx \
      --checkpoint checkpoint/exp/best.pth \
      --variant brecq \
      --quant-state checkpoint/ir50_w4a4_brecq.pth \
      --onnx out/ir50_brecq_fake_quant.onnx

  python -m quantization.run_tensorrt export-onnx \
      --checkpoint checkpoint/exp/best.pth \
      --variant qdq_int8 \
      --quant-state checkpoint/ir50_w4a4_brecq.pth \
      --quant-report out/ir50_brecq_qdq_int8_report.json \
      --onnx out/ir50_brecq_qdq_int8.onnx

  python -m quantization.run_tensorrt build-engine \
      --onnx out/ir50_brecq_qdq_int8.onnx \
      --engine out/ir50_brecq_qdq_int8.engine \
      --explicit-quantization \
      --opt-batch 128

  python -m quantization.run_tensorrt benchmark-pytorch \
      --checkpoint checkpoint/exp/best.pth \
      --variant fp32 \
      --batch-size 128

  python -m quantization.run_tensorrt benchmark-trt \
      --engine out/ir50_fp32_fp16.engine \
      --batch-size 128

Notes:
  - The `brecq` export path preserves the repo's fake-quant graph. It is useful
    for deployment/runtime experiments, but it is not a true INT4 TensorRT path.
  - Real INT4 speedups need a Q/DQ-compatible export or custom TensorRT plugins.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _require_module(name: str) -> None:
    if importlib.util.find_spec(name) is None:
        raise RuntimeError(
            f"Required Python module '{name}' is not installed in the current environment."
        )


def _resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _load_model_for_variant(args, device: str):
    from quantization.exporter import (
        load_brecq_qdq_ir50_export_bundle,
        load_brecq_ir50_classifier,
        load_fp32_ir50_classifier,
    )

    if args.variant == "fp32":
        return load_fp32_ir50_classifier(args.checkpoint, device=device)
    if args.variant == "qdq_int8":
        if not args.quant_state:
            raise ValueError("--quant-state is required for variant='qdq_int8'")
        model, _ = load_brecq_qdq_ir50_export_bundle(
            args.checkpoint,
            args.quant_state,
            device=device,
        )
        return model
    if not args.quant_state:
        raise ValueError("--quant-state is required for variant='brecq'")
    return load_brecq_ir50_classifier(
        args.checkpoint,
        args.quant_state,
        device=device,
    )


def _build_valid_loader(
    checkpoint_path: str | Path,
    dataset_name: str,
    dataset_path: str,
    batch_size: int,
    num_workers: int,
    img_size: int | None = None,
):
    from quantization.data import build_fer_loader
    from quantization.exporter import load_ir50_model_params

    params = load_ir50_model_params(checkpoint_path)
    resolved_img_size = int(img_size or params.get("img_size", 224))
    num_classes = int(params.get("num_classes", 7))

    valid_loader = build_fer_loader(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        img_size=resolved_img_size,
        shuffle=False,
    )
    return valid_loader, num_classes, resolved_img_size


def _infer_img_size(checkpoint_path: str | Path) -> int:
    from quantization.exporter import load_ir50_model_params

    params = load_ir50_model_params(checkpoint_path)
    return int(params.get("img_size", 224))


def _benchmark_pytorch_model(model, input_shape, warmup: int, iters: int):
    import torch

    device = next(model.parameters(), None)
    if device is None:
        device = next(model.buffers()).device
    else:
        device = device.device

    dummy = torch.randn(*input_shape, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                model(dummy)
            end.record()
            torch.cuda.synchronize(device)
            ms_total = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                model(dummy)
            ms_total = (time.perf_counter() - t0) * 1000.0

    ms_per_batch = ms_total / iters
    imgs_per_sec = input_shape[0] / (ms_per_batch / 1000.0)
    return ms_per_batch, imgs_per_sec


def _macro_correct_per_class(logits, labels, num_classes: int):
    import torch

    preds = logits.argmax(dim=1)
    result = torch.zeros(num_classes, device=logits.device, dtype=torch.float32)
    for cls_idx in range(num_classes):
        mask = labels == cls_idx
        if mask.any():
            result[cls_idx] = (preds[mask] == labels[mask]).sum()
    return result


def _evaluate_pytorch_model(model, valid_loader, num_classes: int, device: str, desc: str):
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_macro = torch.zeros(num_classes, device=device, dtype=torch.float32)
    n_items = 0
    per_cls_counts = torch.tensor(
        valid_loader.dataset.get_img_num_per_cls(),
        device=device,
        dtype=torch.float32,
    )

    with torch.no_grad():
        for img, label in tqdm(valid_loader, desc=desc):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logits = model(img)

            loss = F.cross_entropy(logits, label)
            total_loss += loss.item() * label.size(0)
            total_correct += (logits.argmax(dim=1) == label).sum().item()
            total_macro += _macro_correct_per_class(logits, label, num_classes)
            n_items += label.size(0)

    acc = total_correct / n_items
    avg_loss = total_loss / n_items
    macro = (total_macro / per_cls_counts).mean().item()
    return acc, avg_loss, macro


def _trt_dtype_to_torch(dtype):
    import tensorrt as trt
    import torch

    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int8: torch.int8,
        trt.int32: torch.int32,
        trt.bool: torch.bool,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


def _evaluate_trt_engine(engine_path: str | Path, valid_loader, num_classes: int, desc: str):
    import tensorrt as trt
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f:
        engine_bytes = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    if not hasattr(engine, "num_io_tensors"):
        raise RuntimeError(
            "eval-trt requires TensorRT's named tensor API (num_io_tensors). "
            "Please use TensorRT 8.5+ Python bindings."
        )

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context.")
    if not hasattr(context, "execute_async_v3"):
        raise RuntimeError(
            "eval-trt requires execute_async_v3 support in the TensorRT Python API."
        )

    input_names = []
    output_names = []
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)

    if len(input_names) != 1 or len(output_names) != 1:
        raise RuntimeError(
            f"eval-trt currently supports 1 input / 1 output. "
            f"Got inputs={input_names}, outputs={output_names}."
        )

    input_name = input_names[0]
    output_name = output_names[0]

    total_loss = 0.0
    total_correct = 0
    total_macro = torch.zeros(num_classes, device="cuda", dtype=torch.float32)
    n_items = 0
    per_cls_counts = torch.tensor(
        valid_loader.dataset.get_img_num_per_cls(),
        device="cuda",
        dtype=torch.float32,
    )

    stream = torch.cuda.current_stream()
    stream_handle = stream.cuda_stream

    with torch.no_grad():
        for img, label in tqdm(valid_loader, desc=desc):
            img = img.cuda(non_blocking=True).contiguous()
            label = label.cuda(non_blocking=True)

            context.set_input_shape(input_name, tuple(img.shape))
            output_shape = tuple(context.get_tensor_shape(output_name))
            if any(dim < 0 for dim in output_shape):
                output_shape = (img.shape[0], num_classes)

            output_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(output_name))
            logits = torch.empty(output_shape, device="cuda", dtype=output_dtype)

            context.set_tensor_address(input_name, int(img.data_ptr()))
            context.set_tensor_address(output_name, int(logits.data_ptr()))

            ok = context.execute_async_v3(stream_handle)
            if not ok:
                raise RuntimeError("TensorRT execution failed during eval-trt.")
            stream.synchronize()

            logits = logits.float()
            loss = F.cross_entropy(logits, label)
            total_loss += loss.item() * label.size(0)
            total_correct += (logits.argmax(dim=1) == label).sum().item()
            total_macro += _macro_correct_per_class(logits, label, num_classes)
            n_items += label.size(0)

    acc = total_correct / n_items
    avg_loss = total_loss / n_items
    macro = (total_macro / per_cls_counts).mean().item()
    return acc, avg_loss, macro


def _collect_trt_io_names(engine) -> tuple[list[str], list[str]]:
    import tensorrt as trt

    input_names = []
    output_names = []
    for idx in range(engine.num_io_tensors):
        name = engine.get_tensor_name(idx)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            input_names.append(name)
        elif mode == trt.TensorIOMode.OUTPUT:
            output_names.append(name)
    return input_names, output_names


def _load_trt_engine(engine_path: str | Path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f:
        engine_bytes = f.read()
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
    if not hasattr(engine, "num_io_tensors"):
        raise RuntimeError(
            "TensorRT Python fallback requires the named tensor API (num_io_tensors). "
            "Please use TensorRT 8.5+ Python bindings."
        )
    return engine


def _maybe_find_trtexec(explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path
    return shutil.which("trtexec")


def _find_trtexec(explicit_path: str | None) -> str:
    if explicit_path:
        return explicit_path
    found = shutil.which("trtexec")
    if found:
        return found
    raise RuntimeError(
        "trtexec was not found. Pass --trtexec /path/to/trtexec or add it to PATH."
    )


def _format_trt_parser_errors(parser) -> str:
    errors = []
    for idx in range(parser.num_errors):
        errors.append(str(parser.get_error(idx)))
    return "\n".join(errors)


def _build_engine_with_python_trt(args) -> tuple[Path, tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int, int, int]]:
    _require_module("tensorrt")

    import tensorrt as trt

    onnx_path = Path(args.onnx)
    engine_path = Path(args.engine)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError(
            f"TensorRT ONNX parse failed for {onnx_path}.\n{_format_trt_parser_errors(parser)}"
        )

    if network.num_inputs != 1:
        raise RuntimeError(
            f"Python TensorRT builder currently supports a single input tensor, got {network.num_inputs}."
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_mb) << 20)

    if not args.explicit_quantization:
        if args.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif args.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)

    if args.best:
        print("[build-engine] warning: --best is trtexec-specific and is ignored by the Python TensorRT fallback.")
    if args.timing_cache:
        print("[build-engine] warning: --timing-cache is not wired for the Python TensorRT fallback and was ignored.")

    img_size = args.img_size
    min_shape = (args.min_batch, 3, img_size, img_size)
    opt_shape = (args.opt_batch, 3, img_size, img_size)
    max_shape = (args.max_batch, 3, img_size, img_size)

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    input_shape = tuple(input_tensor.shape)
    static_batch = input_shape[0] if len(input_shape) > 0 else None
    requested_batches = {args.min_batch, args.opt_batch, args.max_batch}
    if static_batch is not None and static_batch >= 0 and requested_batches != {static_batch}:
        raise RuntimeError(
            "The ONNX input batch dimension is static "
            f"({input_shape}), but the requested TensorRT profile batches are "
            f"{min_shape}/{opt_shape}/{max_shape}. Re-export the ONNX with "
            "--dynamic-batch, or build with min/opt/max all equal to the fixed batch size."
        )
    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    resolved_shapes = tuple(profile.get_shape(input_tensor.name))
    if resolved_shapes != (min_shape, opt_shape, max_shape):
        raise RuntimeError(
            f"Failed to set TensorRT optimization profile for input {input_tensor.name} "
            f"with shapes {min_shape}/{opt_shape}/{max_shape}. "
            f"TensorRT stored {resolved_shapes} instead. "
            "If the ONNX was exported with a fixed batch size, re-export it with "
            "--dynamic-batch or keep min/opt/max identical."
        )
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT Python build failed: build_serialized_network returned None.")

    engine_path.write_bytes(bytes(serialized))
    return engine_path, min_shape, opt_shape, max_shape


def _benchmark_engine_with_python_trt(engine_path: str | Path, shape: tuple[int, int, int, int], warmup: int, iters: int) -> dict[str, float]:
    _require_module("torch")
    _require_module("tensorrt")

    import torch

    engine = _load_trt_engine(engine_path)
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context.")
    if not hasattr(context, "execute_async_v3"):
        raise RuntimeError(
            "TensorRT Python fallback requires execute_async_v3 support in the TensorRT Python API."
        )

    input_names, output_names = _collect_trt_io_names(engine)
    if len(input_names) != 1 or len(output_names) != 1:
        raise RuntimeError(
            f"Python TensorRT benchmark currently supports 1 input / 1 output. "
            f"Got inputs={input_names}, outputs={output_names}."
        )

    input_name = input_names[0]
    output_name = output_names[0]

    input_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(input_name))
    output_dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(output_name))

    dummy = torch.randn(*shape, device="cuda", dtype=torch.float32).to(input_dtype).contiguous()
    context.set_input_shape(input_name, tuple(dummy.shape))
    output_shape = tuple(context.get_tensor_shape(output_name))
    if any(dim < 0 for dim in output_shape):
        raise RuntimeError(
            f"Failed to resolve TensorRT output shape for {output_name}: {output_shape}"
        )
    output = torch.empty(output_shape, device="cuda", dtype=output_dtype)

    context.set_tensor_address(input_name, int(dummy.data_ptr()))
    context.set_tensor_address(output_name, int(output.data_ptr()))

    stream = torch.cuda.current_stream()
    stream_handle = stream.cuda_stream

    for _ in range(warmup):
        ok = context.execute_async_v3(stream_handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed during warmup.")
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        ok = context.execute_async_v3(stream_handle)
        if not ok:
            raise RuntimeError("TensorRT execution failed during benchmark.")
    end.record()
    torch.cuda.synchronize()

    ms_total = start.elapsed_time(end)
    ms_per_batch = ms_total / iters
    throughput = shape[0] / (ms_per_batch / 1000.0)
    return {
        "latency_ms": ms_per_batch,
        "throughput_qps": throughput,
    }


def _shape_flag(shape: tuple[int, int, int, int]) -> str:
    return "input:" + "x".join(str(v) for v in shape)


def _parse_trtexec_metrics(output: str) -> dict[str, float]:
    metrics: dict[str, float] = {}

    throughput = re.search(r"Throughput:\s*([0-9.]+)\s*qps", output)
    latency = re.search(
        r"Latency:\s*min = [0-9.]+ ms,\s*max = [0-9.]+ ms,\s*mean = ([0-9.]+) ms",
        output,
    )
    enqueue = re.search(
        r"Enqueue Time:\s*min = [0-9.]+ ms,\s*max = [0-9.]+ ms,\s*mean = ([0-9.]+) ms",
        output,
    )

    if throughput:
        metrics["throughput_qps"] = float(throughput.group(1))
    if latency:
        metrics["latency_ms"] = float(latency.group(1))
    if enqueue:
        metrics["enqueue_ms"] = float(enqueue.group(1))

    return metrics


def _run_trtexec(cmd: list[str]) -> tuple[str, dict[str, float]]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + "\n" + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"trtexec failed with code {proc.returncode}.\n{output}")
    return output, _parse_trtexec_metrics(output)


def cmd_export_onnx(args) -> None:
    _require_module("torch")
    _require_module("onnx")

    import torch
    from quantization.exporter import export_ir50_onnx, load_brecq_qdq_ir50_export_bundle

    device = _resolve_device(args.device)
    qdq_report = None
    if args.variant == "qdq_int8":
        model, qdq_report = load_brecq_qdq_ir50_export_bundle(
            args.checkpoint,
            args.quant_state,
            device=device,
        )
    else:
        model = _load_model_for_variant(args, device=device)
    img_size = args.img_size or _infer_img_size(args.checkpoint)
    input_shape = (args.batch_size, 3, img_size, img_size)

    export_ir50_onnx(
        model,
        output_path=args.onnx,
        input_shape=input_shape,
        dynamic_batch=args.dynamic_batch,
        opset=args.opset,
    )

    print(f"[export-onnx] exported {args.variant} model to {args.onnx}")
    print(f"[export-onnx] input shape: {input_shape}  dynamic_batch={args.dynamic_batch}")
    if args.variant == "brecq":
        print("[export-onnx] note: this is the repo's fake-quant graph, not a true INT4 TensorRT graph.")
    elif args.variant == "qdq_int8":
        print("[export-onnx] exported explicit Q/DQ INT8 graph for TensorRT.")
        print("[export-onnx] low-bit layers keep their learned clipping ranges, but ONNX tensor type is INT8.")
        if args.quant_report:
            report_path = Path(args.quant_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(qdq_report, indent=2) + "\n")
            print(f"[export-onnx] wrote quantization report to {report_path}")


def cmd_build_engine(args) -> None:
    trtexec = _maybe_find_trtexec(args.trtexec)
    metrics: dict[str, float] = {}
    output = ""

    if trtexec:
        onnx_path = Path(args.onnx)
        engine_path = Path(args.engine)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        img_size = args.img_size
        min_shape = (args.min_batch, 3, img_size, img_size)
        opt_shape = (args.opt_batch, 3, img_size, img_size)
        max_shape = (args.max_batch, 3, img_size, img_size)

        cmd = [
            trtexec,
            f"--onnx={onnx_path}",
            f"--saveEngine={engine_path}",
            f"--minShapes={_shape_flag(min_shape)}",
            f"--optShapes={_shape_flag(opt_shape)}",
            f"--maxShapes={_shape_flag(max_shape)}",
            f"--workspace={args.workspace_mb}",
            f"--warmUp={args.warmup}",
            f"--iterations={args.iters}",
            "--duration=0",
        ]

        if not args.explicit_quantization:
            if args.precision == "fp16":
                cmd.append("--fp16")
            elif args.precision == "int8":
                cmd.append("--int8")

        if args.use_cuda_graph:
            cmd.append("--useCudaGraph")
        if args.best:
            cmd.append("--best")
        if args.timing_cache:
            cmd.append(f"--timingCacheFile={args.timing_cache}")

        output, metrics = _run_trtexec(cmd)
        backend = "trtexec"
    else:
        engine_path, min_shape, opt_shape, max_shape = _build_engine_with_python_trt(args)
        backend = "python_tensorrt"

    print(f"[build-engine] built TensorRT engine at {engine_path}")
    print(f"[build-engine] backend={backend}")
    mode = "explicit_qdq" if args.explicit_quantization else args.precision
    print(f"[build-engine] precision={mode} profiles={min_shape}/{opt_shape}/{max_shape}")
    if metrics:
        if "throughput_qps" in metrics:
            print(f"[build-engine] throughput={metrics['throughput_qps']:.2f} qps")
        if "latency_ms" in metrics:
            print(f"[build-engine] latency={metrics['latency_ms']:.3f} ms")

    if args.explicit_quantization:
        print("[build-engine] explicit Q/DQ mode: build precision flags were intentionally omitted.")
    elif args.precision == "int8":
        print("[build-engine] warning: INT8 from this ONNX path only helps if the graph exports TRT-compatible quantization ops.")
    if output and args.verbose_output:
        print(output)


def cmd_benchmark_pytorch(args) -> None:
    _require_module("torch")

    device = _resolve_device(args.device)
    model = _load_model_for_variant(args, device=device)
    img_size = args.img_size or _infer_img_size(args.checkpoint)
    input_shape = (args.batch_size, 3, img_size, img_size)

    ms, ips = _benchmark_pytorch_model(
        model,
        input_shape=input_shape,
        warmup=args.warmup,
        iters=args.iters,
    )

    print(f"[benchmark-pytorch] variant={args.variant} device={device}")
    print(f"[benchmark-pytorch] latency={ms:.3f} ms/batch  throughput={ips:.1f} img/s")


def cmd_benchmark_trt(args) -> None:
    img_size = args.img_size
    shape = (args.batch_size, 3, img_size, img_size)

    trtexec = _maybe_find_trtexec(args.trtexec)
    output = ""
    if trtexec:
        cmd = [
            trtexec,
            f"--loadEngine={args.engine}",
            f"--shapes={_shape_flag(shape)}",
            f"--warmUp={args.warmup}",
            f"--iterations={args.iters}",
            "--duration=0",
            "--noDataTransfers",
        ]
        if args.use_cuda_graph:
            cmd.append("--useCudaGraph")

        output, metrics = _run_trtexec(cmd)
        backend = "trtexec"
    else:
        metrics = _benchmark_engine_with_python_trt(
            args.engine,
            shape=shape,
            warmup=args.warmup,
            iters=args.iters,
        )
        backend = "python_tensorrt"

    print(f"[benchmark-trt] engine={args.engine}")
    print(f"[benchmark-trt] backend={backend}")
    if "latency_ms" in metrics:
        print(f"[benchmark-trt] latency={metrics['latency_ms']:.3f} ms/batch")
    if "throughput_qps" in metrics:
        print(f"[benchmark-trt] throughput={metrics['throughput_qps']:.1f} img/s")
    if "enqueue_ms" in metrics:
        print(f"[benchmark-trt] enqueue={metrics['enqueue_ms']:.3f} ms")
    if output and not metrics:
        print(output)
    elif output and args.verbose_output:
        print(output)


def cmd_eval_pytorch(args) -> None:
    _require_module("torch")

    device = _resolve_device(args.device)
    valid_loader, num_classes, img_size = _build_valid_loader(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    model = _load_model_for_variant(args, device=device)
    acc, loss, macro = _evaluate_pytorch_model(
        model,
        valid_loader=valid_loader,
        num_classes=num_classes,
        device=device,
        desc=f"[eval-pytorch:{args.variant}]",
    )

    print(
        f"[eval-pytorch] variant={args.variant} dataset={args.dataset_name} "
        f"img_size={img_size} device={device}"
    )
    print(f"[eval-pytorch] acc={acc:.4f}  macro_acc={macro:.4f}  loss={loss:.4f}")


def cmd_eval_trt(args) -> None:
    _require_module("torch")
    _require_module("tensorrt")

    valid_loader, num_classes, img_size = _build_valid_loader(
        checkpoint_path=args.checkpoint,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )
    acc, loss, macro = _evaluate_trt_engine(
        args.engine,
        valid_loader=valid_loader,
        num_classes=num_classes,
        desc="[eval-trt]",
    )

    print(f"[eval-trt] engine={args.engine} dataset={args.dataset_name} img_size={img_size}")
    print(f"[eval-trt] acc={acc:.4f}  macro_acc={macro:.4f}  loss={loss:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IR50 ONNX/TensorRT deployment helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    export_p = sub.add_parser("export-onnx", help="Export FP32 or BRECQ IR50 to ONNX")
    export_p.add_argument("--checkpoint", required=True, help="Training checkpoint with backbone.* and weight")
    export_p.add_argument("--variant", choices=("fp32", "brecq", "qdq_int8"), required=True)
    export_p.add_argument("--quant-state", default=None, help="BRECQ state file (.pth). Required for variant=brecq")
    export_p.add_argument("--onnx", required=True, help="Output ONNX path")
    export_p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    export_p.add_argument("--batch-size", type=int, default=1)
    export_p.add_argument("--img-size", type=int, default=None, help="Override input image size")
    export_p.add_argument("--dynamic-batch", action="store_true")
    export_p.add_argument("--opset", type=int, default=17)
    export_p.add_argument("--quant-report", default=None,
                          help="Optional JSON path for per-layer Q/DQ export report (qdq_int8 only)")
    export_p.set_defaults(func=cmd_export_onnx)

    build_p = sub.add_parser("build-engine", help="Build TensorRT engine from ONNX using trtexec")
    build_p.add_argument("--onnx", required=True)
    build_p.add_argument("--engine", required=True)
    build_p.add_argument("--trtexec", default=None, help="Path to trtexec")
    build_p.add_argument("--precision", choices=("fp32", "fp16", "int8"), default="fp16")
    build_p.add_argument("--explicit-quantization", action="store_true",
                         help="Build a TensorRT engine from an explicit Q/DQ ONNX model without precision flags")
    build_p.add_argument("--img-size", type=int, default=224)
    build_p.add_argument("--min-batch", type=int, default=1)
    build_p.add_argument("--opt-batch", type=int, default=64)
    build_p.add_argument("--max-batch", type=int, default=128)
    build_p.add_argument("--workspace-mb", type=int, default=4096)
    build_p.add_argument("--warmup", type=int, default=50)
    build_p.add_argument("--iters", type=int, default=200)
    build_p.add_argument("--timing-cache", default=None)
    build_p.add_argument("--use-cuda-graph", action="store_true")
    build_p.add_argument("--best", action="store_true")
    build_p.add_argument("--verbose-output", action="store_true")
    build_p.set_defaults(func=cmd_build_engine)

    torch_p = sub.add_parser("benchmark-pytorch", help="Benchmark PyTorch FP32 or BRECQ model")
    torch_p.add_argument("--checkpoint", required=True)
    torch_p.add_argument("--variant", choices=("fp32", "brecq", "qdq_int8"), required=True)
    torch_p.add_argument("--quant-state", default=None)
    torch_p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    torch_p.add_argument("--batch-size", type=int, default=128)
    torch_p.add_argument("--img-size", type=int, default=None)
    torch_p.add_argument("--warmup", type=int, default=20)
    torch_p.add_argument("--iters", type=int, default=100)
    torch_p.set_defaults(func=cmd_benchmark_pytorch)

    trt_p = sub.add_parser("benchmark-trt", help="Benchmark a TensorRT engine with trtexec")
    trt_p.add_argument("--engine", required=True)
    trt_p.add_argument("--trtexec", default=None, help="Path to trtexec")
    trt_p.add_argument("--batch-size", type=int, default=128)
    trt_p.add_argument("--img-size", type=int, default=224)
    trt_p.add_argument("--warmup", type=int, default=50)
    trt_p.add_argument("--iters", type=int, default=200)
    trt_p.add_argument("--use-cuda-graph", action="store_true")
    trt_p.add_argument("--verbose-output", action="store_true")
    trt_p.set_defaults(func=cmd_benchmark_trt)

    eval_torch_p = sub.add_parser("eval-pytorch", help="Evaluate a PyTorch deployment wrapper on a validation set")
    eval_torch_p.add_argument("--checkpoint", required=True)
    eval_torch_p.add_argument("--variant", choices=("fp32", "brecq", "qdq_int8"), required=True)
    eval_torch_p.add_argument("--quant-state", default=None)
    eval_torch_p.add_argument("--dataset-name", choices=("RAF-DB", "AffectNet", "CAER"), required=True)
    eval_torch_p.add_argument("--dataset-path", required=True)
    eval_torch_p.add_argument("--device", default="auto", help="auto | cuda | cpu")
    eval_torch_p.add_argument("--batch-size", type=int, default=128)
    eval_torch_p.add_argument("--num-workers", type=int, default=0)
    eval_torch_p.add_argument("--img-size", type=int, default=None)
    eval_torch_p.set_defaults(func=cmd_eval_pytorch)

    eval_trt_p = sub.add_parser("eval-trt", help="Evaluate a TensorRT engine on a validation set")
    eval_trt_p.add_argument("--checkpoint", required=True,
                            help="Training checkpoint used to recover dataset/model metadata")
    eval_trt_p.add_argument("--engine", required=True)
    eval_trt_p.add_argument("--dataset-name", choices=("RAF-DB", "AffectNet", "CAER"), required=True)
    eval_trt_p.add_argument("--dataset-path", required=True)
    eval_trt_p.add_argument("--batch-size", type=int, default=128)
    eval_trt_p.add_argument("--num-workers", type=int, default=0)
    eval_trt_p.add_argument("--img-size", type=int, default=None)
    eval_trt_p.set_defaults(func=cmd_eval_trt)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"[run_tensorrt] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
