from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint_tag", type=str, default="best", choices=["best", "last"])
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    return parser.parse_args()


def resolve_checkpoint_dir(checkpoint_dir: str, checkpoint_tag: str) -> Path:
    root = Path(checkpoint_dir).expanduser().resolve()
    tagged = root / checkpoint_tag
    if tagged.is_dir():
        return tagged
    return root


def load_rank_state(checkpoint_dir: Path) -> dict:
    rank0 = checkpoint_dir / "train_state.r0.pt"
    if not rank0.is_file():
        raise FileNotFoundError(f"Missing rank state: {rank0}")
    rank_state = torch.load(rank0, map_location="cpu", weights_only=False)
    if not isinstance(rank_state, dict):
        raise TypeError(f"Invalid rank state object in {rank0}")
    return rank_state


def build_runtime_args(checkpoint_args: dict) -> argparse.Namespace:
    runtime = argparse.Namespace()
    runtime.architecture = str(checkpoint_args.get("architecture", "kprpe_base"))
    runtime.embedding_dim = int(checkpoint_args.get("embedding_dim", 512))
    runtime.use_flash_attn = bool(checkpoint_args.get("use_flash_attn", True))
    runtime.rpe_impl = str(checkpoint_args.get("rpe_impl", "extension"))
    return runtime


def detect_fsdp_local_state(
    state_dict: dict[str, torch.Tensor],
    target_state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str]]:
    zero_sized: list[str] = []
    flattened: list[str] = []
    for key, tensor in state_dict.items():
        if key not in target_state_dict or not torch.is_tensor(tensor):
            continue
        expected = target_state_dict[key]
        if tensor.numel() == 0 and expected.numel() > 0:
            zero_sized.append(key)
            continue
        if tensor.numel() == expected.numel() and tuple(tensor.shape) != tuple(expected.shape):
            flattened.append(key)
    return zero_sized, flattened


def validate_checkpoint_state_dict(model_path: Path, runtime_args: argparse.Namespace) -> None:
    probe_model = get_model(runtime_args)
    target_state_dict = probe_model.state_dict()
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    zero_sized_keys, flattened_keys = detect_fsdp_local_state(
        state_dict=state_dict,
        target_state_dict=target_state_dict,
    )
    if zero_sized_keys:
        raise ValueError(
            "Checkpoint contains zero-sized parameter shards and is not a full model state_dict. "
            f"Example keys: {zero_sized_keys[:5]}. "
            "This is an old rank0-only FSDP save and cannot be exported to single-GPU eval from model.pt alone."
        )
    if flattened_keys:
        raise ValueError(
            "Checkpoint contains flattened local FSDP parameter tensors instead of full parameter shapes. "
            f"Example keys: {flattened_keys[:5]}. "
            "This old checkpoint format cannot be exported to single-GPU eval from model.pt alone."
        )


def export_single_process(model_path: Path, output_path: Path, runtime_args: argparse.Namespace, device: torch.device) -> None:
    model = get_model(runtime_args).to(device)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def export_accelerated(
    model_path: Path,
    output_path: Path,
    runtime_args: argparse.Namespace,
    mixed_precision: str,
    expected_world_size: int,
) -> None:
    from accelerate import Accelerator, FullyShardedDataParallelPlugin

    accelerator = Accelerator(
        mixed_precision=mixed_precision if mixed_precision in {"no", "fp16", "bf16"} else "no",
        fsdp_plugin=FullyShardedDataParallelPlugin(use_orig_params=True),
    )
    if accelerator.num_processes != expected_world_size:
        raise ValueError(
            f"World size mismatch for FSDP export: launched={accelerator.num_processes}, "
            f"checkpoint={expected_world_size}. Launch with matching torchrun --nproc_per_node."
        )
    model = get_model(runtime_args)
    model = accelerator.prepare(model)

    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    accelerator.unwrap_model(model).load_state_dict(state_dict, strict=True)
    full_state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(full_state_dict, output_path)
        print(f"[EXPORT] saved full model state_dict to {output_path}")
    accelerator.wait_for_everyone()


def main():
    args = parse_args()
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint_dir, args.checkpoint_tag)
    rank_state = load_rank_state(checkpoint_dir)
    checkpoint_args = rank_state.get("args", {})
    if not isinstance(checkpoint_args, dict):
        raise TypeError(f"Invalid checkpoint args in {checkpoint_dir / 'train_state.r0.pt'}")

    model_path = checkpoint_dir / "model.pt"
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {model_path}")

    output_path = Path(args.output_path).expanduser().resolve() if args.output_path else checkpoint_dir / "model.exported.pt"
    if output_path.exists() and not args.overwrite:
        print(f"[EXPORT] reuse existing exported model: {output_path}")
        return

    runtime_args = build_runtime_args(checkpoint_args)
    use_accelerator = bool(checkpoint_args.get("use_accelerator", False))
    validate_checkpoint_state_dict(model_path=model_path, runtime_args=runtime_args)

    if use_accelerator:
        export_accelerated(
            model_path=model_path,
            output_path=output_path,
            runtime_args=runtime_args,
            mixed_precision=str(checkpoint_args.get("mixed_precision", "no")),
            expected_world_size=int(checkpoint_args.get("world_size", 1)),
        )
        return

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    export_single_process(model_path=model_path, output_path=output_path, runtime_args=runtime_args, device=device)
    print(f"[EXPORT] saved full model state_dict to {output_path}")


if __name__ == "__main__":
    main()
