#!/usr/bin/env python3
import argparse
import os
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

# Optional image backends
try:
    from PIL import Image  # type: ignore
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import imageio.v2 as imageio  # type: ignore
    IMAGEIO_AVAILABLE = True
except Exception:
    IMAGEIO_AVAILABLE = False

SUPPORTED_FORMATS = {"png", "jpg", "jpeg"}


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def find_npz(path: Path, name_hint: str) -> Optional[Path]:
    """Return a path to an .npz in directory containing name_hint, else None."""
    candidates = [p for p in path.glob("*.npz") if name_hint in p.name]
    if not candidates:
        return None
    # Prefer exact match first
    for p in candidates:
        if p.name == f"{name_hint}.npz":
            return p
    # Otherwise first by name sort
    return sorted(candidates)[0]


def extract_npy_from_npz(npz_path: Path, tmp_dir: Path) -> Tuple[Path, Path]:
    """Extract arr_0.npy (images) and arr_1.npy (labels) from npz into tmp_dir.

    Returns tuple of (images_npy_path, labels_npy_path)
    """
    with zipfile.ZipFile(npz_path, 'r') as zf:
        names = zf.namelist()
        # Heuristics: arrays stored as arr_0.npy (images) and arr_1.npy (labels)
        if 'arr_0.npy' not in names or 'arr_1.npy' not in names:
            raise RuntimeError(
                f"Unexpected .npz structure in {npz_path.name}: expected 'arr_0.npy' and 'arr_1.npy', found {names}")
        img_out = tmp_dir / 'arr_0.npy'
        lbl_out = tmp_dir / 'arr_1.npy'
        # Extract to temp dir
        zf.extract('arr_0.npy', path=tmp_dir)
        zf.extract('arr_1.npy', path=tmp_dir)
        return img_out, lbl_out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_image_writer(image_format: str):
    fmt = image_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format '{image_format}'. Choose from {sorted(SUPPORTED_FORMATS)}")

    def write_with_pil(img_array: np.ndarray, dst: Path, quality: int = 95):
        im = Image.fromarray(img_array)
        if fmt in {"jpg", "jpeg"}:
            im.save(dst, format='JPEG', quality=quality, optimize=True)
        else:
            im.save(dst, format='PNG', optimize=True)

    def write_with_imageio(img_array: np.ndarray, dst: Path, quality: int = 95):
        if fmt in {"jpg", "jpeg"}:
            imageio.imwrite(dst, img_array, quality=quality)
        else:
            imageio.imwrite(dst, img_array)

    if PIL_AVAILABLE:
        return write_with_pil
    if IMAGEIO_AVAILABLE:
        return write_with_imageio
    raise RuntimeError("No image backend available. Install Pillow (pip install pillow) or imageio (pip install imageio).")


def save_images(
    images_mm: np.memmap,
    labels_mm: np.memmap,
    indices: np.ndarray,
    split_name: str,
    out_root: Path,
    image_format: str,
    quality: int,
    workers: int,
) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    writer = get_image_writer(image_format)
    fmt = image_format.lower()

    def task(sample_index: int) -> Optional[Path]:
        img = images_mm[sample_index]
        lbl = int(labels_mm[sample_index])
        if not (0 <= lbl <= 13):
            return None
        split_dir = out_root / split_name / str(lbl)
        ensure_dir(split_dir)
        dst = split_dir / f"{sample_index:07d}.{fmt if fmt != 'jpg' else 'jpg'}"
        writer(img, dst, quality)
        return dst

    if workers <= 1:
        for idx in indices:
            task(int(idx))
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(task, int(idx)) for idx in indices]
        # Consume futures to surface exceptions
        for fut in as_completed(futures):
            _ = fut.result()


def main():
    parser = argparse.ArgumentParser(
        description="Convert Clothing1M .npz to foldered image dataset (train/valid/0-13). Uses all of clothing1m.npz for train and any *test*.npz for valid.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "\n" \
            "Requirements:\n" \
            "  - Place 'clothing1m.npz' and one or more '*test*.npz' files (e.g., 'clothing10k_test.npz') in the input directory.\n" \
            "  - Install an image backend: 'pip install pillow' or 'pip install imageio'.\n" \
            "\n" \
            "Basic usage:\n" \
            "  python convert_clothing1m_to_images.py .\n" \
            "\n" \
            "Common options:\n" \
            "  --output-dir DIR         Output root (default: ./images_clothing1m)\n" \
            "  --image-format {png,jpg,jpeg}   Image format (default: png)\n" \
            "  --quality N              JPEG quality 1-100 (default: 95)\n" \
            "  --workers N              Parallel writer threads (default: 8)\n" \
            "  --limit N                Limit samples per split for quick runs\n" \
            "\n" \
            "Examples:\n" \
            "  python convert_clothing1m_to_images.py . --output-dir ./images_clothing1m\n" \
            "  python convert_clothing1m_to_images.py . --image-format jpg --quality 90\n" \
            "  python convert_clothing1m_to_images.py . --workers 16 --limit 10000\n" \
            "\n" \
            "What it does:\n" \
            "  - Uses ALL samples from 'clothing1m.npz' for the training split.\n" \
            "  - Uses ALL samples from every '*test*.npz' for the validation split.\n" \
            "  - Creates directories: train/0..13 and valid/0..13 with images.\n" \
            "  - '--limit' applies independently to train and to the TOTAL across all valid files.\n" \
            "\n" \
            "Notes:\n" \
            "  - The script errors if no '*test*.npz' is found in the input directory.\n" \
            "  - Writing many images is I/O heavy; SSD + higher '--workers' speeds it up.\n+            "
        ),
    )
    parser.add_argument("input_dir", type=str, help="Directory containing clothing1m .npz files")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: <input_dir>/images_clothing1m)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--image-format", type=str, default="png", choices=sorted(SUPPORTED_FORMATS), help="Image format to save")
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel writer threads")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples per split (for quick tests)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        eprint(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (input_dir / "images_clothing1m")
    ensure_dir(output_dir)

    # Locate datasets
    one_m_npz = find_npz(input_dir, "clothing1m")
    if one_m_npz is None:
        eprint("Could not find clothing1m .npz in input directory")
        sys.exit(1)

    # Find all test npz files (e.g., *test*.npz)
    test_npz_files: List[Path] = sorted([p for p in input_dir.glob("*.npz") if "test" in p.name])

    # Prepare temp area for extracted .npy files
    tmp_root = Path(tempfile.mkdtemp(prefix="clothing1m_npz_extract_", dir=str(output_dir)))

    try:
        eprint(f"Extracting arrays from {one_m_npz.name} to {tmp_root} ...")
        images_npy, labels_npy = extract_npy_from_npz(one_m_npz, tmp_root)

        eprint("Memory-mapping arrays (this creates large .npy files on disk if not already extracted)...")
        images = np.load(images_npy, mmap_mode='r')
        labels = np.load(labels_npy, mmap_mode='r')

        if images.ndim != 4 or images.shape[3] not in (1, 3, 4):
            raise RuntimeError(f"Unexpected image array shape {images.shape}. Expected (N,H,W,C)")
        if labels.ndim != 1 or labels.shape[0] != images.shape[0]:
            raise RuntimeError(f"Labels shape {labels.shape} does not match images {images.shape}")

        num_samples = images.shape[0]
        eprint(f"Dataset: {num_samples} images of size {images.shape[1]}x{images.shape[2]}x{images.shape[3]}")

        # Create class directories
        for split in ("train", "valid"):
            for c in range(14):
                ensure_dir(output_dir / split / str(c))

        # Always use the entire clothing1m.npz for training
        train_indices = np.arange(num_samples)
        if args.limit:
            train_indices = train_indices[: args.limit]

        eprint(f"Writing train images ({len(train_indices)}) ...")
        save_images(images, labels, train_indices, "train", output_dir, args.image_format, args.quality, args.workers)

        # Use any *test*.npz for validation. If multiple, concatenate.
        if not test_npz_files:
            eprint("Validation set '*test*.npz' not found in input directory.")
            sys.exit(1)

        all_valid_images: List[np.ndarray] = []
        all_valid_labels: List[np.ndarray] = []
        total_valid = 0
        for tnpz in test_npz_files:
            eprint(f"Extracting arrays from {tnpz.name} to {tmp_root} ...")
            v_images_npy, v_labels_npy = extract_npy_from_npz(tnpz, tmp_root)
            v_images = np.load(v_images_npy, mmap_mode='r')
            v_labels = np.load(v_labels_npy, mmap_mode='r')
            all_valid_images.append(v_images)
            all_valid_labels.append(v_labels)
            total_valid += v_images.shape[0]

        # If only one file, use it directly; otherwise create a virtual index space spanning all
        if len(all_valid_images) == 1:
            v_images = all_valid_images[0]
            v_labels = all_valid_labels[0]
            valid_indices = np.arange(v_images.shape[0])
            if args.limit:
                valid_indices = valid_indices[: args.limit]
            eprint(f"Writing valid images ({len(valid_indices)}) ...")
            save_images(v_images, v_labels, valid_indices, "valid", output_dir, args.image_format, args.quality, args.workers)
        else:
            # Concatenate by iterating file-by-file to avoid huge memory copies
            # Respect --limit across the union
            remaining = args.limit if args.limit is not None else None
            from concurrent.futures import ThreadPoolExecutor, as_completed
            writer = get_image_writer(args.image_format)
            fmt = args.image_format.lower()

            def per_file_write(images_mm: np.memmap, labels_mm: np.memmap, indices: np.ndarray):
                def task(sample_index: int):
                    img = images_mm[sample_index]
                    lbl = int(labels_mm[sample_index])
                    if not (0 <= lbl <= 13):
                        return
                    dst = output_dir / "valid" / str(lbl) / f"{sample_index:07d}.{fmt if fmt != 'jpg' else 'jpg'}"
                    ensure_dir(dst.parent)
                    writer(img, dst, args.quality)
                if args.workers <= 1:
                    for idx in indices:
                        task(int(idx))
                else:
                    with ThreadPoolExecutor(max_workers=args.workers) as ex:
                        futures = [ex.submit(task, int(i)) for i in indices]
                        for fut in as_completed(futures):
                            _ = fut.result()

            written = 0
            for v_images, v_labels in zip(all_valid_images, all_valid_labels):
                count = v_images.shape[0]
                idx = np.arange(count)
                if remaining is not None:
                    take = max(0, min(remaining - written, count))
                    idx = idx[: take]
                per_file_write(v_images, v_labels, idx)
                written += idx.shape[0]
                if remaining is not None and written >= remaining:
                    break

        eprint("Done.")

    finally:
        # Keep extracted .npy files to avoid re-extraction if output_dir is persistent? For safety, clean up.
        try:
            shutil.rmtree(tmp_root)
        except Exception:
            pass


if __name__ == "__main__":
    main()
