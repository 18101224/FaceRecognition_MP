#!/usr/bin/env python3
"""
run_brecq.py — Entry point for BRECQ post-training quantisation of IR50.

Example:
    # W4A4, 1024 calibration images from RAF-DB train split
    python -m quantization.run_brecq \
        --checkpoint  checkpoint/ir50.pth \
        --calib_dir   /data/RAF-DB/train \
        --n_calib     1024 \
        --output      checkpoint/ir50_w4a4_brecq.pth \
        --w_bits 4 --a_bits 4 \
        --n_iters 20000 \
        --use_fisher

    # Quick smoke-test (W8A8, 64 samples, 200 iters)
    python -m quantization.run_brecq \
        --checkpoint checkpoint/ir50.pth \
        --calib_dir  /data/RAF-DB/train \
        --n_calib 64 --n_iters 200 \
        --w_bits 8 --a_bits 8 \
        --output /tmp/ir50_test.pth
"""

import argparse
import os
import random
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Allow running as `python -m quantization.run_brecq` from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.modules.ir50 import Backbone
from quantization import BRECQOptimizer


# ---------------------------------------------------------------------------
# Calibration dataset
# ---------------------------------------------------------------------------

class CalibDataset(Dataset):
    """Flat directory of face images for calibration.

    Walks ``root`` recursively, collects up to ``n_samples`` image paths,
    and applies the standard IR50 pre-processing pipeline.
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, n_samples: int = 1024,
                 img_size: int = 112, seed: int = 42):
        super().__init__()
        all_paths = []
        for dirpath, _, fnames in os.walk(root):
            for fn in fnames:
                if os.path.splitext(fn)[1].lower() in self.EXTENSIONS:
                    all_paths.append(os.path.join(dirpath, fn))

        if not all_paths:
            raise FileNotFoundError(f"No images found in {root!r}")

        rng = random.Random(seed)
        rng.shuffle(all_paths)
        self.paths = all_paths[:n_samples]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # ImageNet-style normalisation; adjust mean/std if your IR50
            # was trained with a different pipeline.
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), 0   # label unused


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

@torch.no_grad()
def cosine_sim_degradation(optimizer: BRECQOptimizer,
                           calib_loader: DataLoader,
                           device: torch.device) -> float:
    """Measure mean cosine similarity between FP32 and W4A4 embeddings
    on the calibration set.  Values close to 1.0 mean minimal degradation."""
    fp_model = optimizer._fp_model.eval()

    sims = []
    for imgs, _ in calib_loader:
        imgs = imgs.to(device)
        imgs_112 = F.interpolate(imgs, size=112)

        # FP32 embedding
        with torch.no_grad():
            x = fp_model.input_layer(imgs_112)
            x = fp_model.body1(x)
            x = fp_model.body2(x)
            x = fp_model.body3(x)
            fp_emb = F.adaptive_avg_pool2d(x, 1).reshape(imgs.size(0), -1)
            fp_emb = F.normalize(fp_emb, dim=1)

        # Quantised embedding
        q_emb = optimizer.forward(imgs)
        q_emb = F.normalize(q_emb, dim=1)

        sim = (fp_emb * q_emb).sum(dim=1).mean().item()
        sims.append(sim)

    return sum(sims) / len(sims)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(
        description="BRECQ PTQ for IR50 face-recognition backbone"
    )
    # Model & data
    p.add_argument("--checkpoint",  default="checkpoint/ir50.pth",
                   help="Path to the pretrained IR50 .pth file")
    p.add_argument("--calib_dir",   required=True,
                   help="Directory of calibration face images (walked recursively)")
    p.add_argument("--n_calib",     type=int, default=1024,
                   help="Number of calibration images (default 1024)")
    p.add_argument("--output",      default="checkpoint/ir50_brecq.pth",
                   help="Path to save the quantised model")

    # Quantisation settings
    p.add_argument("--w_bits",          type=int, default=4,
                   help="Weight bit-width for body Conv layers (default 4)")
    p.add_argument("--a_bits",          type=int, default=4,
                   help="Activation bit-width (default 4)")
    p.add_argument("--first_last_bits", type=int, default=8,
                   help="Bit-width for input_layer (default 8)")

    # BRECQ hyper-parameters
    p.add_argument("--n_iters",    type=int,   default=20_000,
                   help="BRECQ optimisation iterations per block (default 20000)")
    p.add_argument("--lam",        type=float, default=1e-2,
                   help="AdaRound regularisation weight λ (default 0.01)")
    p.add_argument("--batch_size", type=int,   default=32,
                   help="Mini-batch size for BRECQ inner loop (default 32)")
    p.add_argument("--use_fisher", action="store_true",
                   help="Use Fisher-information weighting in reconstruction loss")
    p.add_argument("--opt_target", choices=["both", "weights", "activations"],
                   default="both",
                   help="Optimise weight rounding vars, activation scales, or both")
    p.add_argument("--reg_reduction", choices=["sum", "mean"], default="sum",
                   help="Reduction for AdaRound regularisation across weights")
    p.add_argument("--act_init_mode", choices=["lsq", "max", "percentile"],
                   default="lsq",
                   help="Activation scale initialisation rule")
    p.add_argument("--act_init_percentile", type=float, default=0.999,
                   help="Percentile used when act_init_mode=percentile")
    p.add_argument("--act_init_samples", type=int, default=64,
                   help="Calibration samples used to seed activation scales")

    # Runtime
    p.add_argument("--device",      default="cuda",
                   help="Device string: cuda | cuda:0 | cpu (default cuda)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--eval",        action="store_true",
                   help="After quantisation, report cosine-similarity degradation")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[run_brecq] Device: {device}")

    # ---- Load FP32 model ----
    print(f"[run_brecq] Loading IR50 from {args.checkpoint}")
    fp_model = Backbone(args.checkpoint)
    fp_model.eval()

    # ---- Calibration loader ----
    print(f"[run_brecq] Building calibration dataset from {args.calib_dir!r} "
          f"(n={args.n_calib})")
    calib_ds = CalibDataset(args.calib_dir, n_samples=args.n_calib)
    calib_loader = DataLoader(
        calib_ds, batch_size=64, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ---- BRECQ ----
    optimizer = BRECQOptimizer(
        fp_model,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
        first_last_bits=args.first_last_bits,
        n_iters=args.n_iters,
        lam=args.lam,
        batch_size=args.batch_size,
        use_fisher=args.use_fisher,
        opt_target=args.opt_target,
        reg_reduction=args.reg_reduction,
        act_init_mode=args.act_init_mode,
        act_init_percentile=args.act_init_percentile,
        act_init_samples=args.act_init_samples,
        device=str(device),
        verbose=True,
    )

    optimizer.quantize(calib_loader)

    # ---- Save ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    optimizer.save(args.output)

    # ---- Optional evaluation ----
    if args.eval:
        print("\n[run_brecq] Evaluating cosine-similarity degradation on calibration set…")
        sim = cosine_sim_degradation(optimizer, calib_loader, device)
        print(f"[run_brecq] Mean cosine similarity (FP32 vs W{args.w_bits}A{args.a_bits}): "
              f"{sim:.4f}  (1.0 = identical)")


if __name__ == "__main__":
    main()
