from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import sklearn
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from aligners import get_aligner
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
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--aligner_ckpt", type=str, default=None)
    parser.add_argument("--eval_root", type=str, default="/data/mj/facerec_val")
    parser.add_argument("--datasets", nargs="+", default=["lfw", "agedb_30", "cfp_fp"])
    parser.add_argument("--architecture", type=str, default=None)
    parser.add_argument("--embedding_dim", type=int, default=None)
    parser.add_argument("--use_flash_attn", type=str2bool, default=None)
    parser.add_argument("--rpe_impl", type=str, default="auto", choices=["auto", "extension", "triton"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--flip_test", type=str2bool, default=True)
    parser.add_argument("--n_folds", type=int, default=10)
    return parser.parse_args()


def resolve_model_path(args: argparse.Namespace) -> Path:
    if args.model_path:
        model_path = Path(args.model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"model_path does not exist: {model_path}")
        return model_path

    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve()
    if checkpoint_dir.is_file():
        if checkpoint_dir.name != "model.pt":
            raise ValueError(f"Expected model.pt file, got: {checkpoint_dir}")
        return checkpoint_dir

    tagged = checkpoint_dir / args.checkpoint_tag / "model.pt"
    if tagged.is_file():
        return tagged

    direct = checkpoint_dir / "model.pt"
    if direct.is_file():
        return direct

    raise FileNotFoundError(
        f"Could not resolve model.pt from checkpoint_dir={checkpoint_dir}, checkpoint_tag={args.checkpoint_tag}"
    )


def resolve_train_state_path(model_path: Path) -> Path | None:
    candidate_dirs = [model_path.parent]
    if model_path.parent.name not in {"best", "last"}:
        candidate_dirs.extend([model_path.parent / "best", model_path.parent / "last"])
    for checkpoint_dir in candidate_dirs:
        rank0 = checkpoint_dir / "train_state.r0.pt"
        if rank0.is_file():
            return rank0
        ranked = sorted(checkpoint_dir.glob("train_state.r*.pt"))
        if ranked:
            return ranked[0]
    return None


def load_checkpoint_args(model_path: Path) -> Dict[str, object]:
    rank_state_path = resolve_train_state_path(model_path)
    if rank_state_path is None:
        return {}
    rank_state = torch.load(rank_state_path, map_location="cpu", weights_only=False)
    args = rank_state.get("args", {})
    if not isinstance(args, dict):
        return {}
    return args


def resolve_rpe_impl(args: argparse.Namespace, checkpoint_args: Dict[str, object]) -> str:
    ckpt_impl_obj = checkpoint_args.get("rpe_impl")
    ckpt_impl = str(ckpt_impl_obj).strip().lower() if ckpt_impl_obj is not None else None
    requested = str(args.rpe_impl).strip().lower()

    if requested == "auto":
        if ckpt_impl is None:
            raise ValueError(
                "Could not infer rpe_impl from checkpoint metadata. "
                "Pass --rpe_impl explicitly to avoid mixing train/eval RPE paths."
            )
        return ckpt_impl

    if ckpt_impl is not None and requested != ckpt_impl:
        raise ValueError(
            f"rpe_impl mismatch: eval arg={requested}, checkpoint={ckpt_impl}. "
            "Use the checkpoint backend to avoid mixed train/eval RPE paths."
        )
    return requested


def resolve_eval_dataset(path: Path):
    ds = load_from_disk(str(path))
    if isinstance(ds, Dataset):
        return ds
    if isinstance(ds, DatasetDict):
        if len(ds) == 0:
            raise ValueError(f"Empty DatasetDict at {path}")
        split = next(iter(ds.keys()))
        return ds[split]
    raise TypeError(f"Unsupported dataset object from {path}: {type(ds)!r}")


def build_collate_fn(image_transform):
    def _collate(examples):
        images = [image_transform(example["image"].convert("RGB")) for example in examples]
        indexes = [int(example["index"]) for example in examples]
        is_same = [bool(example["is_same"]) for example in examples]
        return (
            torch.stack(images, dim=0),
            torch.tensor(indexes, dtype=torch.long),
            torch.tensor(is_same, dtype=torch.bool),
        )

    return _collate


def get_autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda":
        return nullcontext()
    if mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def calculate_accuracy(threshold: float, dist: np.ndarray, actual_issame: np.ndarray) -> Tuple[float, float, float]:
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0.0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0.0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / float(max(dist.size, 1))
    return tpr, fpr, acc


def evaluate_verification(embeddings: np.ndarray, issame_pairs: np.ndarray, n_folds: int = 10) -> Dict[str, float]:
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    if embeddings1.shape[0] != embeddings2.shape[0]:
        raise ValueError(
            f"Invalid embeddings shape for pair eval: {embeddings.shape}, "
            "expected even number of samples."
        )

    diff = embeddings1 - embeddings2
    dist = np.sum(np.square(diff), axis=1)
    thresholds = np.arange(0.0, 4.0, 0.01)
    n_pairs = min(len(issame_pairs), len(dist))
    indices = np.arange(n_pairs)

    k_fold = KFold(n_splits=n_folds, shuffle=False)
    accuracies = np.zeros((n_folds,), dtype=np.float64)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        train_acc = np.zeros((len(thresholds),), dtype=np.float64)
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, train_acc[threshold_idx] = calculate_accuracy(
                threshold,
                dist[train_set],
                issame_pairs[train_set],
            )
        best_threshold = thresholds[np.argmax(train_acc)]
        _, _, accuracies[fold_idx] = calculate_accuracy(best_threshold, dist[test_set], issame_pairs[test_set])

    return {
        "acc": float(np.mean(accuracies) * 100.0),
        "std": float(np.std(accuracies) * 100.0),
    }


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    aligner: torch.nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: str,
    flip_images: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()
    if aligner is not None:
        aligner.eval()

    all_embeddings: List[torch.Tensor] = []
    all_indexes: List[torch.Tensor] = []
    all_is_same: List[torch.Tensor] = []

    for images, indexes, is_same in loader:
        images = images.to(device, non_blocking=True)
        if flip_images:
            images = torch.flip(images, dims=[3])

        keypoints = None
        if aligner is not None:
            _, _, keypoints, _, _, _ = aligner(images)
            keypoints = keypoints.to(device, non_blocking=True)

        with get_autocast_context(device, mixed_precision):
            embeddings = model(images, keypoints=keypoints) if keypoints is not None else model(images)

        all_embeddings.append(embeddings.detach().cpu())
        all_indexes.append(indexes.detach().cpu())
        all_is_same.append(is_same.detach().cpu())

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    indexes = torch.cat(all_indexes, dim=0).numpy()
    is_same = torch.cat(all_is_same, dim=0).numpy()

    sort_order = np.argsort(indexes)
    return {
        "embeddings": embeddings[sort_order],
        "indexes": indexes[sort_order],
        "is_same": is_same[sort_order],
    }


def build_runtime_args(args: argparse.Namespace, rpe_impl: str) -> argparse.Namespace:
    runtime = argparse.Namespace()
    runtime.architecture = args.architecture
    runtime.embedding_dim = args.embedding_dim
    runtime.use_flash_attn = args.use_flash_attn
    runtime.rpe_impl = rpe_impl
    return runtime


def resolve_runtime_value(
    cli_value,
    checkpoint_args: Dict[str, object],
    key: str,
    default,
):
    if cli_value is not None:
        return cli_value
    checkpoint_value = checkpoint_args.get(key)
    if checkpoint_value is not None:
        return checkpoint_value
    return default


def detect_fsdp_local_state(
    state_dict: Dict[str, torch.Tensor],
    target_state_dict: Dict[str, torch.Tensor],
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


def main():
    args = parse_args()
    model_path = resolve_model_path(args)
    checkpoint_args = load_checkpoint_args(model_path)
    args.architecture = str(resolve_runtime_value(args.architecture, checkpoint_args, "architecture", "kprpe_base"))
    args.embedding_dim = int(resolve_runtime_value(args.embedding_dim, checkpoint_args, "embedding_dim", 512))
    args.use_flash_attn = bool(resolve_runtime_value(args.use_flash_attn, checkpoint_args, "use_flash_attn", True))
    resolved_rpe_impl = resolve_rpe_impl(args=args, checkpoint_args=checkpoint_args)

    device = torch.device(args.device) if args.device else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    runtime_args = build_runtime_args(args, rpe_impl=resolved_rpe_impl)
    model = get_model(runtime_args).to(device)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
    zero_sized_keys, flattened_keys = detect_fsdp_local_state(state_dict=state_dict, target_state_dict=model.state_dict())
    if zero_sized_keys:
        raise ValueError(
            "Checkpoint appears to contain FSDP local/sharded parameters instead of a full model state_dict. "
            f"Example zero-sized keys: {zero_sized_keys[:5]}. "
            "This checkpoint cannot be evaluated with single-file loading. Re-export a full state_dict first."
        )
    if flattened_keys:
        raise ValueError(
            "Checkpoint appears to contain flattened parameter tensors instead of full parameter shapes. "
            f"Example flattened keys: {flattened_keys[:5]}. "
            "Re-export a full state_dict before running verification eval."
        )
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    needs_keypoints = str(args.architecture).startswith("kprpe")
    aligner = None
    if needs_keypoints:
        if not args.aligner_ckpt:
            raise ValueError("--aligner_ckpt is required for kprpe evaluation.")
        aligner = get_aligner(args.aligner_ckpt).to(device)
        aligner.eval()

    eval_root = Path(args.eval_root).expanduser().resolve()
    transform = model.make_test_transform()
    collate_fn = build_collate_fn(transform)

    print(f"[INFO] model_path={model_path}")
    print(f"[INFO] device={device} mixed_precision={args.mixed_precision} flip_test={args.flip_test}")
    print(f"[INFO] rpe_impl={resolved_rpe_impl}")
    print(f"[INFO] eval_root={eval_root}")

    results = {}
    for dataset_name in args.datasets:
        dataset_path = eval_root / dataset_name
        if not dataset_path.exists():
            print(f"[WARN] skip {dataset_name}: not found at {dataset_path}")
            continue

        dataset = resolve_eval_dataset(dataset_path)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_fn,
            drop_last=False,
        )

        collection = extract_features(
            model=model,
            aligner=aligner,
            loader=loader,
            device=device,
            mixed_precision=args.mixed_precision,
            flip_images=False,
        )

        embeddings = collection["embeddings"]
        if args.flip_test:
            collection_flip = extract_features(
                model=model,
                aligner=aligner,
                loader=loader,
                device=device,
                mixed_precision=args.mixed_precision,
                flip_images=True,
            )
            embeddings = embeddings + collection_flip["embeddings"]

        embeddings = sklearn.preprocessing.normalize(embeddings)
        issame_pairs = collection["is_same"][::2]
        metric = evaluate_verification(embeddings=embeddings, issame_pairs=issame_pairs, n_folds=args.n_folds)
        results[dataset_name] = metric
        print(f"[RESULT] {dataset_name}: acc={metric['acc']:.4f} std={metric['std']:.4f}")

    if not results:
        print("[INFO] No datasets evaluated.")
        return

    mean_acc = float(np.mean([item["acc"] for item in results.values()]))
    print(f"[SUMMARY] mean_acc={mean_acc:.4f} over {len(results)} datasets")


if __name__ == "__main__":
    main()
