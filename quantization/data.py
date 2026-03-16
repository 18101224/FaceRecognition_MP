"""Minimal FER dataset helpers for quantization workflows."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


FER_DATASET_NAMES = ("RAF-DB", "AffectNet", "CAER")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _build_eval_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _resolve_split(dataset_name: str, train: bool) -> str:
    if train:
        return "train"
    if dataset_name == "CAER":
        return "test"
    return "valid"


class FERDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_path: str | Path,
        train: bool,
        img_size: int,
    ):
        super().__init__()
        if dataset_name not in FER_DATASET_NAMES:
            raise ValueError(
                f"Unsupported dataset '{dataset_name}'. Supported: {', '.join(FER_DATASET_NAMES)}"
            )

        self.dataset_name = dataset_name
        self.root = Path(dataset_path)
        self.transform = _build_eval_transform(img_size)
        self.paths: list[Path] = []
        self.labels: list[int] = []

        split_dir = self.root / _resolve_split(dataset_name, train)
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_dir}")

        if dataset_name == "CAER":
            class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
            for class_idx, class_dir in enumerate(class_dirs):
                image_paths = sorted(
                    path
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
                self.paths.extend(image_paths)
                self.labels.extend([class_idx] * len(image_paths))
        else:
            label_offset = 1 if dataset_name == "RAF-DB" else 0
            for class_idx in range(7):
                class_dir = split_dir / str(class_idx + label_offset)
                if not class_dir.exists():
                    raise FileNotFoundError(f"Expected class directory not found: {class_dir}")
                image_paths = sorted(
                    path
                    for path in class_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                )
                self.paths.extend(image_paths)
                self.labels.extend([class_idx] * len(image_paths))

        if not self.paths:
            raise FileNotFoundError(f"No images found under {split_dir}")

        self._img_num_per_cls = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]

    def get_img_num_per_cls(self):
        if self._img_num_per_cls is None:
            counter = Counter(self.labels)
            self._img_num_per_cls = [counter.get(idx, 0) for idx in range(max(counter) + 1)]
        return list(self._img_num_per_cls)


class RecursiveImageDataset(Dataset):
    def __init__(self, root: str | Path, img_size: int):
        super().__init__()
        root = Path(root)
        self.paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found under {root}")
        self.transform = _build_eval_transform(img_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), 0


def build_fer_loader(
    dataset_name: str,
    dataset_path: str | Path,
    train: bool,
    batch_size: int,
    num_workers: int,
    img_size: int,
    shuffle: bool = False,
):
    dataset = FERDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        train=train,
        img_size=img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader


def build_calibration_loader(
    dataset_name: str,
    dataset_path: str | Path,
    batch_size: int,
    num_workers: int,
    img_size: int,
    calib_ratio: float = 1.0,
    seed: int = 42,
):
    dataset = FERDataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        train=True,
        img_size=img_size,
    )
    if not 0 < calib_ratio <= 1.0:
        raise ValueError(f"calib_ratio must be in (0, 1], got {calib_ratio}")
    if calib_ratio < 1.0:
        n_items = max(1, int(len(dataset) * calib_ratio))
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=generator)[:n_items].tolist()
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader, len(dataset)
