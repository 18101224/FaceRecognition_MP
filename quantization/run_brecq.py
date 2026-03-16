#!/usr/bin/env python3
"""Run BRECQ PTQ for an IR50 FER classifier checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

from quantization import BRECQOptimizer
from quantization.data import (
    FER_DATASET_NAMES,
    RecursiveImageDataset,
    build_calibration_loader,
    build_fer_loader,
)
from quantization.exporter import load_fp32_ir50_classifier, load_ir50_model_params


@torch.no_grad()
def _macro_correct_per_class(logits, labels, num_classes: int):
    preds = logits.argmax(dim=1)
    result = torch.zeros(num_classes, device=logits.device, dtype=torch.float32)
    for cls_idx in range(num_classes):
        mask = labels == cls_idx
        if mask.any():
            result[cls_idx] = (preds[mask] == labels[mask]).sum()
    return result


@torch.no_grad()
def _validate_classifier(model, valid_loader, num_classes: int, device: str, desc: str):
    from tqdm import tqdm

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_macro = torch.zeros(num_classes, device=device, dtype=torch.float32)
    total_items = 0
    per_cls_counts = torch.tensor(
        valid_loader.dataset.get_img_num_per_cls(),
        device=device,
        dtype=torch.float32,
    )

    for img, label in tqdm(valid_loader, desc=desc):
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model(img)
        loss = F.cross_entropy(logits, label)
        total_loss += loss.item() * label.size(0)
        total_correct += (logits.argmax(dim=1) == label).sum().item()
        total_macro += _macro_correct_per_class(logits, label, num_classes)
        total_items += label.size(0)

    return (
        total_correct / total_items,
        total_loss / total_items,
        (total_macro / per_cls_counts).mean().item(),
    )


@torch.no_grad()
def _validate_embeddings(embed_fn, classifier_weight, valid_loader, num_classes: int, device: str, desc: str):
    from tqdm import tqdm

    total_loss = 0.0
    total_correct = 0
    total_macro = torch.zeros(num_classes, device=device, dtype=torch.float32)
    total_items = 0
    per_cls_counts = torch.tensor(
        valid_loader.dataset.get_img_num_per_cls(),
        device=device,
        dtype=torch.float32,
    )
    weight = F.normalize(classifier_weight.to(device), dim=0)

    for img, label in tqdm(valid_loader, desc=desc):
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        emb = embed_fn(img)
        emb = F.normalize(emb, dim=-1, eps=1e-6)
        logits = emb @ weight

        loss = F.cross_entropy(logits, label)
        total_loss += loss.item() * label.size(0)
        total_correct += (logits.argmax(dim=1) == label).sum().item()
        total_macro += _macro_correct_per_class(logits, label, num_classes)
        total_items += label.size(0)

    return (
        total_correct / total_items,
        total_loss / total_items,
        (total_macro / per_cls_counts).mean().item(),
    )


@torch.no_grad()
def _prime_quant_act_scales(brecq_opt, calib_loader, n_samples: int = 64):
    parts = []
    collected = 0
    for batch in calib_loader:
        imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
        imgs = imgs.cpu()
        if imgs.shape[-2:] != (112, 112):
            imgs = F.interpolate(imgs, size=112)
        need = n_samples - collected
        if need <= 0:
            break
        parts.append(imgs[:need])
        collected += min(need, imgs.shape[0])
        if collected >= n_samples:
            break

    if not parts:
        raise RuntimeError("No calibration images were collected to initialize activation quantizers.")

    x = torch.cat(parts, dim=0).to(brecq_opt.device)
    for block in brecq_opt._quant_blocks:
        block.init_act_quantizers(x)
        x = block(x)


@torch.no_grad()
def _measure_latency(forward_fn, input_shape, device: str, warmup: int = 20, iters: int = 100):
    dummy = torch.randn(*input_shape, device=device)

    for _ in range(warmup):
        forward_fn(dummy)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            forward_fn(dummy)
        end.record()
        torch.cuda.synchronize()
        ms_total = start.elapsed_time(end)
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            forward_fn(dummy)
        ms_total = (time.perf_counter() - t0) * 1000.0

    ms_per_batch = ms_total / iters
    imgs_per_sec = input_shape[0] / (ms_per_batch / 1000.0)
    return ms_per_batch, imgs_per_sec


def _build_calib_loader(args, img_size: int):
    from torch.utils.data import DataLoader

    if args.calib_dir:
        dataset = RecursiveImageDataset(args.calib_dir, img_size=img_size)
        if args.n_calib:
            dataset.paths = dataset.paths[: args.n_calib]
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        return loader, len(dataset)

    if not args.dataset_name or not args.dataset_path:
        raise ValueError("Either --calib-dir or both --dataset-name and --dataset-path are required.")

    return build_calibration_loader(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
        calib_ratio=args.calib_ratio,
        seed=args.seed,
    )


def _run_eval_mode(args, img_size: int, num_classes: int, device: str):
    if not args.dataset_name or not args.dataset_path:
        raise ValueError("Evaluation requires --dataset-name and --dataset-path.")
    valid_loader = build_fer_loader(
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        train=False,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        img_size=img_size,
        shuffle=False,
    )
    model = load_fp32_ir50_classifier(args.checkpoint, device=device)
    return valid_loader, model


def build_parser():
    p = argparse.ArgumentParser(description="BRECQ PTQ for IR50 FER checkpoints")
    p.add_argument("--checkpoint", "--ckpt_path", dest="checkpoint", required=True)
    p.add_argument("--dataset-name", "--dataset_name", dest="dataset_name", choices=FER_DATASET_NAMES, default=None)
    p.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", default=None)
    p.add_argument("--calib-dir", "--calib_dir", dest="calib_dir", default=None)
    p.add_argument("--n-calib", "--n_calib", dest="n_calib", type=int, default=None)
    p.add_argument("--output", "--quant-output", "--quant_output", dest="output", required=True)
    p.add_argument("--w-bits", "--w_bits", dest="w_bits", type=int, default=4)
    p.add_argument("--a-bits", "--a_bits", dest="a_bits", type=int, default=8)
    p.add_argument("--first-last-bits", "--first_last_bits", dest="first_last_bits", type=int, default=8)
    p.add_argument("--n-iters", "--n_iters", dest="n_iters", type=int, default=20_000)
    p.add_argument("--lam", type=float, default=1e-4)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=32)
    p.add_argument("--eval-batch-size", "--eval_batch_size", dest="eval_batch_size", type=int, default=128)
    p.add_argument("--use-fisher", "--use_fisher", dest="use_fisher", action="store_true")
    p.add_argument("--precompute", action="store_true")
    p.add_argument("--opt-target", "--opt_target", dest="opt_target", choices=("both", "weights", "activations"), default="both")
    p.add_argument("--reg-reduction", "--reg_reduction", dest="reg_reduction", choices=("sum", "mean"), default="mean")
    p.add_argument("--act-init-mode", "--act_init_mode", dest="act_init_mode", choices=("lsq", "max", "percentile"), default="percentile")
    p.add_argument("--act-init-percentile", "--act_init_percentile", dest="act_init_percentile", type=float, default=0.999)
    p.add_argument("--act-init-samples", "--act_init_samples", dest="act_init_samples", type=int, default=256)
    p.add_argument("--calib-ratio", "--calib_ratio", dest="calib_ratio", type=float, default=0.1)
    p.add_argument("--img-size", "--img_size", dest="img_size", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-eval", "--skip_eval", dest="skip_eval", action="store_true")
    p.add_argument("--model-type", "--model_type", dest="model_type", default="ir50")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.model_type != "ir50":
        raise ValueError(f"Only model_type='ir50' is supported, got {args.model_type!r}")

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    params = load_ir50_model_params(args.checkpoint)
    img_size = int(args.img_size or params.get("img_size", 224))
    num_classes = int(params.get("num_classes", 7))

    fp_model = load_fp32_ir50_classifier(args.checkpoint, device=device)
    calib_loader, n_calib = _build_calib_loader(args, img_size=img_size)

    print(f"[run_brecq] device={device}")
    print(f"[run_brecq] checkpoint={args.checkpoint}")
    if args.dataset_name and args.dataset_path:
        print(f"[run_brecq] dataset={args.dataset_name} path={args.dataset_path}")
    elif args.calib_dir:
        print(f"[run_brecq] calib_dir={args.calib_dir}")
    print(f"[run_brecq] calibration_items={n_calib}")

    brecq = BRECQOptimizer(
        fp_model.backbone,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        first_last_bits=args.first_last_bits,
        n_iters=args.n_iters,
        lam=args.lam,
        batch_size=args.batch_size,
        use_fisher=args.use_fisher,
        precompute=args.precompute,
        opt_target=args.opt_target,
        reg_reduction=args.reg_reduction,
        act_init_mode=args.act_init_mode,
        act_init_percentile=args.act_init_percentile,
        act_init_samples=args.act_init_samples,
        device=device,
        verbose=True,
    )

    if args.skip_eval:
        brecq.quantize(calib_loader)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        brecq.save(str(output_path))
        return

    valid_loader, eval_model = _run_eval_mode(args, img_size=img_size, num_classes=num_classes, device=device)

    fp_acc, fp_loss, fp_macro = _validate_classifier(
        eval_model,
        valid_loader=valid_loader,
        num_classes=num_classes,
        device=device,
        desc="[FP32] Validating",
    )
    fp_ms, fp_ips = _measure_latency(
        lambda x: eval_model(x),
        input_shape=(args.eval_batch_size, 3, img_size, img_size),
        device=device,
    )

    folded_acc, folded_loss, folded_macro = _validate_embeddings(
        lambda x: brecq._fp_model(x)[0],
        eval_model.weight,
        valid_loader=valid_loader,
        num_classes=num_classes,
        device=device,
        desc="[Folded FP32] Validating",
    )

    _prime_quant_act_scales(brecq, calib_loader, n_samples=args.act_init_samples)
    pre_q_acc, pre_q_loss, pre_q_macro = _validate_embeddings(
        brecq.forward,
        eval_model.weight,
        valid_loader=valid_loader,
        num_classes=num_classes,
        device=device,
        desc=f"[Pre-BRECQ W{args.w_bits}A{args.a_bits}] Validating",
    )

    brecq.quantize(calib_loader)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    brecq.save(str(output_path))

    post_q_acc, post_q_loss, post_q_macro = _validate_embeddings(
        brecq.forward,
        eval_model.weight,
        valid_loader=valid_loader,
        num_classes=num_classes,
        device=device,
        desc=f"[W{args.w_bits}A{args.a_bits}] Validating",
    )
    post_q_ms, post_q_ips = _measure_latency(
        lambda x: brecq.forward(x),
        input_shape=(args.eval_batch_size, 3, img_size, img_size),
        device=device,
    )

    print()
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  original_fp32_acc      = {fp_acc:.4f}")
    print(f"  folded_fp32_acc        = {folded_acc:.4f}")
    print(f"  pre_brecq_quant_acc    = {pre_q_acc:.4f}")
    print(f"  post_brecq_quant_acc   = {post_q_acc:.4f}")
    print()
    print(f"  fp32_loss              = {fp_loss:.4f}")
    print(f"  folded_fp32_loss       = {folded_loss:.4f}")
    print(f"  pre_brecq_quant_loss   = {pre_q_loss:.4f}")
    print(f"  post_brecq_quant_loss  = {post_q_loss:.4f}")
    print()
    print(f"  fp32_macro_acc         = {fp_macro:.4f}")
    print(f"  folded_fp32_macro_acc  = {folded_macro:.4f}")
    print(f"  pre_brecq_macro_acc    = {pre_q_macro:.4f}")
    print(f"  post_brecq_macro_acc   = {post_q_macro:.4f}")
    print()
    print(f"  fp32_latency_ms        = {fp_ms:.2f}")
    print(f"  quant_latency_ms       = {post_q_ms:.2f}")
    print(f"  fp32_throughput        = {fp_ips:.0f}")
    print(f"  quant_throughput       = {post_q_ips:.0f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI error path
        print(f"[run_brecq] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
