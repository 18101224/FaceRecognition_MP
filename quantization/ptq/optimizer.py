"""
BRECQ block-wise reconstruction optimiser for IR50.

Two modes controlled by ``precompute``:

  precompute=True  (fast, high memory)
      Pre-store X, Z_fp, W_f on CPU per block.  Mini-batches indexed
      directly → ~2× faster, but peak CPU memory ≈ 76 GB for 12K images
      on early blocks (64×112×112 activations).

  precompute=False  (slow, low memory)
      Only raw calibration images stored on CPU (~1.7 GB for 12K).
      Block inputs recomputed through the frozen quantised prefix each
      iteration.  Fisher compressed to per-channel.  ~2.5× slower but
      peak CPU memory stays ≈ 2 GB.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional

from ..calibration.bn_fold import fold_ir50
from .blocks import build_quant_blocks


_CHUNK = 64  # images per chunk for batched inference


# ---------------------------------------------------------------------------
# Beta annealing schedule
# ---------------------------------------------------------------------------

def _get_beta(cur: int, total: int, warmup: float = 0.2,
              beta_start: float = 20.0, beta_end: float = 2.0) -> float:
    if cur < warmup * total:
        return beta_start
    progress = (cur - warmup * total) / ((1.0 - warmup) * total + 1e-8)
    return beta_start - (beta_start - beta_end) * progress


def _get_reg_coef(cur: int, total: int, warmup: float = 0.2,
                  lam: float = 1e-2) -> float:
    return 0.0 if cur < warmup * total else lam


# ---------------------------------------------------------------------------
# FP32 suffix (blocks i+1 … end + adaptive_avg_pool)
# ---------------------------------------------------------------------------

class _IR50Suffix(nn.Module):
    def __init__(self, folded_model, block_start_body_idx: int):
        super().__init__()
        body_blocks = []
        for bname in ("body1", "body2", "body3"):
            b = getattr(folded_model, bname, None)
            if b is not None:
                body_blocks.extend(list(b))
        self.suffix_blocks = nn.Sequential(*body_blocks[block_start_body_idx:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.suffix_blocks(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.size(0), -1)
        return x


# ---------------------------------------------------------------------------
# Main BRECQ optimiser
# ---------------------------------------------------------------------------

class BRECQOptimizer:
    """Block-wise reconstruction PTQ for IR50.

    Args:
        precompute: If True, precompute and store X / Z_fp / W_f on CPU
            per block (fast, but needs ~76 GB CPU RAM for 12K images on
            early blocks).  If False, recompute through the quantised
            prefix each iteration (~2 GB CPU, ~2.5× slower).
    """

    def __init__(
        self,
        fp_model,
        w_bits: int = 4,
        a_bits: int = 4,
        first_last_bits: int = 8,
        n_iters: int = 20_000,
        lam: float = 1e-2,
        batch_size: int = 32,
        use_fisher: bool = True,
        precompute: bool = True,
        opt_target: str = "both",
        reg_reduction: str = "sum",
        act_init_mode: str = "lsq",
        act_init_percentile: float = 0.999,
        act_init_samples: int = 64,
        device: str = "cuda",
        verbose: bool = True,
    ):
        self.w_bits          = w_bits
        self.a_bits          = a_bits
        self.first_last_bits = first_last_bits
        self.n_iters         = n_iters
        self.lam             = lam
        self.batch_size      = batch_size
        self.use_fisher      = use_fisher
        self.precompute      = precompute
        self.opt_target      = opt_target
        self.reg_reduction   = reg_reduction
        self.act_init_mode   = act_init_mode
        self.act_init_percentile = act_init_percentile
        self.act_init_samples = act_init_samples
        self.device          = torch.device(device)
        self.verbose         = verbose

        if self.opt_target not in {"both", "weights", "activations"}:
            raise ValueError(f"Unsupported opt_target: {self.opt_target}")
        if self.reg_reduction not in {"sum", "mean"}:
            raise ValueError(f"Unsupported reg_reduction: {self.reg_reduction}")
        if self.act_init_samples < 1:
            raise ValueError("act_init_samples must be >= 1")

        # Deep copy + fold BN
        self._fp_model = deepcopy(fp_model).to(self.device).eval()
        fold_ir50(self._fp_model)
        self._fp_model.to(self.device)
        for p in self._fp_model.parameters():
            p.requires_grad_(False)

        # Build quantised blocks (all on GPU)
        self._quant_blocks, self._block_names = build_quant_blocks(
            self._fp_model,
            w_bits=w_bits, a_bits=a_bits,
            first_layer_bits=first_last_bits,
            act_init_mode=act_init_mode,
            act_init_percentile=act_init_percentile,
        )
        for blk in self._quant_blocks:
            blk.to(self.device)

        self._n_body_blocks = sum(
            len(getattr(self._fp_model, bn, []))
            for bn in ("body1", "body2", "body3")
        )

        self._calib_images: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, calib_loader) -> None:
        """Run full BRECQ loop.  calib_loader yields (images, labels)."""
        self._calib_images = self._gather_calib(calib_loader)  # CPU [N,3,112,112]

        if self.verbose:
            gb = (self._calib_images.nelement()
                  * self._calib_images.element_size() / 2**30)
            mode = "precompute" if self.precompute else "on-the-fly"
            print(f"[BRECQ] Calibration set: {self._calib_images.shape[0]} "
                  f"images ({gb:.1f} GB CPU)  mode={mode}")
            print(f"[BRECQ] Quantising {len(self._quant_blocks)} blocks "
                  f"(W{self.w_bits}A{self.a_bits}, "
                  f"first-layer W{self.first_last_bits}A{self.first_last_bits})")

        if self.precompute:
            self._quantize_precompute()
        else:
            self._quantize_onthefly()

    def save(self, path: str) -> None:
        state = {
            "w_bits": self.w_bits,
            "a_bits": self.a_bits,
            "first_last_bits": self.first_last_bits,
            "lam": self.lam,
            "batch_size": self.batch_size,
            "use_fisher": self.use_fisher,
            "precompute": self.precompute,
            "opt_target": self.opt_target,
            "reg_reduction": self.reg_reduction,
            "act_init_mode": self.act_init_mode,
            "act_init_percentile": self.act_init_percentile,
            "act_init_samples": self.act_init_samples,
            "blocks": {
                name: blk.state_dict()
                for name, blk in zip(self._block_names, self._quant_blocks)
            },
        }
        torch.save(state, path)
        if self.verbose:
            print(f"[BRECQ] Saved quantised model to {path}")

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        for name, blk in zip(self._block_names, self._quant_blocks):
            blk.load_state_dict(state["blocks"][name])
        for blk in self._quant_blocks:
            if hasattr(blk, "act_quant"):
                blk.act_quant._initialized = True
            if hasattr(blk, "conv"):
                blk.conv.weight_quantizer._frozen = True
            if hasattr(blk, "conv1"):
                blk.conv1.weight_quantizer._frozen = True
            if hasattr(blk, "conv2"):
                blk.conv2.weight_quantizer._frozen = True
            shortcut = getattr(blk, "shortcut", None)
            if shortcut is not None and hasattr(shortcut, "weight_quantizer"):
                shortcut.weight_quantizer._frozen = True
        if self.verbose:
            print(f"[BRECQ] Loaded quantised model from {path}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full quantised forward (mirrors IR50 Backbone.forward)."""
        x = F.interpolate(x, size=112)
        for blk in self._quant_blocks:
            x = blk(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.size(0), -1)
        return x

    # ==================================================================
    #  MODE 1: precompute  (fast, high CPU memory)
    # ==================================================================

    def _quantize_precompute(self) -> None:
        prefix_acts = self._calib_images          # CPU [N,3,112,112]

        for block_idx, (qblk, name) in enumerate(
                zip(self._quant_blocks, self._block_names)):

            t0 = time.time()
            if self.verbose:
                print(f"\n[BRECQ] Block {block_idx+1}/"
                      f"{len(self._quant_blocks)}: {name}")

            X = prefix_acts                       # CPU [N, C, H, W]

            # FP32 block outputs — CPU
            Z_fp = self._fp_block_outputs(qblk, X)

            if self.verbose:
                nbytes = (X.nelement() * X.element_size()
                          + Z_fp.nelement() * Z_fp.element_size())
                print(f"  stored activations: "
                      f"X {list(X.shape)}  Z_fp {list(Z_fp.shape)}  "
                      f"({nbytes / 2**30:.1f} GB CPU)")

            # Element-wise Fisher weights — CPU
            if self.use_fisher and block_idx < len(self._quant_blocks) - 1:
                W_f = self._fisher_weights(Z_fp, block_idx)
            else:
                W_f = None

            # Seed LSQ
            sample = X[:min(self.act_init_samples, X.shape[0])].to(self.device)
            qblk.init_act_quantizers(sample)
            del sample

            # Optimise
            self._optimize_block_precompute(qblk, X, Z_fp, W_f)

            qblk.freeze()

            # Advance prefix
            prefix_acts = self._run_through_block(qblk, X)

            del X, Z_fp, W_f
            torch.cuda.empty_cache()

            if self.verbose:
                print(f"  done in {time.time() - t0:.1f}s")

        if self.verbose:
            print("\n[BRECQ] All blocks quantised.")

    # -- precompute helpers --

    @torch.no_grad()
    def _fp_block_outputs(self, qblk, X: torch.Tensor) -> torch.Tensor:
        """FP32 block outputs for all X.  Returns CPU tensor."""
        parts = []
        for start in range(0, X.shape[0], _CHUNK):
            chunk = X[start:start + _CHUNK].to(self.device)
            parts.append(qblk.fp_forward(chunk).cpu())
        return torch.cat(parts, dim=0)

    @torch.no_grad()
    def _run_through_block(self, qblk, X: torch.Tensor) -> torch.Tensor:
        """Run quantised block on all X (CPU→GPU→CPU in chunks)."""
        parts = []
        for start in range(0, X.shape[0], _CHUNK):
            chunk = X[start:start + _CHUNK].to(self.device)
            parts.append(qblk(chunk).cpu())
        return torch.cat(parts, dim=0)

    def _fisher_weights(self, Z_fp: torch.Tensor,
                        block_idx: int) -> Optional[torch.Tensor]:
        """Element-wise Fisher weights (grad^2).  Z_fp on CPU → CPU."""
        suffix_start = 0 if block_idx == 0 else block_idx
        if suffix_start >= self._n_body_blocks:
            return None

        suffix = _IR50Suffix(
            self._fp_model, suffix_start).to(self.device).eval()
        for p in suffix.parameters():
            p.requires_grad_(False)

        grads_sq = []
        for start in range(0, Z_fp.shape[0], _CHUNK):
            z = Z_fp[start:start + _CHUNK].to(self.device)
            z = z.detach().requires_grad_(True)
            emb = suffix(z)
            loss = emb.norm(dim=1).mean()
            loss.backward()
            grads_sq.append(z.grad.detach().pow(2).cpu())
            del z, emb, loss
            torch.cuda.empty_cache()

        del suffix
        torch.cuda.empty_cache()
        return torch.cat(grads_sq, dim=0)

    def _optimize_block_precompute(
        self,
        qblk,
        X: torch.Tensor,                   # CPU
        Z_fp: torch.Tensor,                # CPU
        W_f: Optional[torch.Tensor],       # CPU or None
    ) -> None:
        N = X.shape[0]
        ada_params, lsq_params = qblk.opt_params()

        optimizer = torch.optim.Adam(self._build_param_groups(ada_params, lsq_params))

        log_every = max(1, self.n_iters // 10)

        qblk.train()
        for it in range(self.n_iters):
            idx = torch.randint(0, N, (self.batch_size,))

            x_b  = X[idx].to(self.device)
            z_b  = Z_fp[idx].to(self.device)
            wf_b = W_f[idx].to(self.device) if W_f is not None else None

            z_q = qblk(x_b)

            diff_sq = (z_b - z_q).pow(2)
            if wf_b is not None:
                recon = (wf_b * diff_sq).mean()
            else:
                recon = diff_sq.mean()

            beta = _get_beta(it, self.n_iters)
            coef = _get_reg_coef(it, self.n_iters, lam=self.lam)
            reg  = self._adaround_reg_loss(qblk, ada_params, beta, coef)

            loss = recon + reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose and (it + 1) % log_every == 0:
                print(f"  iter {it+1:5d}/{self.n_iters}  "
                      f"recon={recon.item():.4e}  reg={reg.item():.4e}  "
                      f"beta={beta:.1f}")

        qblk.eval()

    # ==================================================================
    #  MODE 2: on-the-fly  (slow, low CPU memory)
    # ==================================================================

    def _quantize_onthefly(self) -> None:
        for block_idx, (qblk, name) in enumerate(
                zip(self._quant_blocks, self._block_names)):

            t0 = time.time()
            if self.verbose:
                print(f"\n[BRECQ] Block {block_idx+1}/"
                      f"{len(self._quant_blocks)}: {name}")

            # Per-channel Fisher [C_out] — tiny
            if self.use_fisher and block_idx < len(self._quant_blocks) - 1:
                fisher_chan = self._channel_fisher(qblk, block_idx)
            else:
                fisher_chan = None

            # Seed LSQ
            n_seed = min(self.act_init_samples, self._calib_images.shape[0])
            with torch.no_grad():
                sample = self._prefix_forward(
                    self._calib_images[:n_seed], block_idx)
            qblk.init_act_quantizers(sample)
            del sample

            # Optimise
            self._optimize_block_onthefly(qblk, block_idx, fisher_chan)

            qblk.freeze()
            torch.cuda.empty_cache()

            if self.verbose:
                print(f"  done in {time.time() - t0:.1f}s")

        if self.verbose:
            print("\n[BRECQ] All blocks quantised.")

    # -- on-the-fly helpers --

    @torch.no_grad()
    def _prefix_forward(self, images: torch.Tensor,
                        block_idx: int) -> torch.Tensor:
        """Run images through quantised prefix (blocks 0 … block_idx-1).

        Args:
            images: CPU tensor [B, 3, 112, 112]
        Returns:
            GPU tensor [B, C, H, W]
        """
        x = images.to(self.device)
        for blk in self._quant_blocks[:block_idx]:
            x = blk(x)
        return x

    def _channel_fisher(self, qblk,
                        block_idx: int) -> Optional[torch.Tensor]:
        """Per-channel Fisher E_n,h,w[(dL/dz)^2].  Returns [C_out] CPU."""
        suffix_start = 0 if block_idx == 0 else block_idx
        if suffix_start >= self._n_body_blocks:
            return None

        suffix = _IR50Suffix(
            self._fp_model, suffix_start).to(self.device).eval()
        for p in suffix.parameters():
            p.requires_grad_(False)

        accum = None
        count = 0
        N = self._calib_images.shape[0]

        for start in range(0, N, _CHUNK):
            chunk = self._calib_images[start:start + _CHUNK]

            with torch.no_grad():
                x = self._prefix_forward(chunk, block_idx)
                z_fp = qblk.fp_forward(x)
            z = z_fp.detach().requires_grad_(True)

            emb = suffix(z)
            loss = emb.norm(dim=1).mean()
            loss.backward()

            g2 = z.grad.detach().pow(2)
            g2 = (g2.mean(dim=(0, 2, 3)) if g2.dim() == 4
                  else g2.mean(dim=0))

            accum = g2.cpu() if accum is None else accum + g2.cpu()
            count += 1

            del x, z_fp, z, emb, loss
            torch.cuda.empty_cache()

        del suffix
        torch.cuda.empty_cache()

        if self.verbose and accum is not None:
            print(f"  Fisher: per-channel weights ({count} chunks)")

        return accum / count if accum is not None else None

    def _optimize_block_onthefly(
        self,
        qblk,
        block_idx: int,
        fisher_chan: Optional[torch.Tensor],
    ) -> None:
        N = self._calib_images.shape[0]
        ada_params, lsq_params = qblk.opt_params()

        optimizer = torch.optim.Adam(self._build_param_groups(ada_params, lsq_params))

        fisher_gpu = (fisher_chan.to(self.device)
                      if fisher_chan is not None else None)

        log_every = max(1, self.n_iters // 10)

        qblk.train()
        for it in range(self.n_iters):
            idx = torch.randint(0, N, (self.batch_size,))
            imgs = self._calib_images[idx]

            with torch.no_grad():
                x_b  = self._prefix_forward(imgs, block_idx)
                z_fp = qblk.fp_forward(x_b)

            z_q = qblk(x_b)

            diff_sq = (z_fp - z_q).pow(2)
            if fisher_gpu is not None:
                if diff_sq.dim() == 4:
                    recon = (diff_sq
                             * fisher_gpu[None, :, None, None]).mean()
                else:
                    recon = (diff_sq * fisher_gpu[None, :]).mean()
            else:
                recon = diff_sq.mean()

            beta = _get_beta(it, self.n_iters)
            coef = _get_reg_coef(it, self.n_iters, lam=self.lam)
            reg  = self._adaround_reg_loss(qblk, ada_params, beta, coef)

            loss = recon + reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose and (it + 1) % log_every == 0:
                print(f"  iter {it+1:5d}/{self.n_iters}  "
                      f"recon={recon.item():.4e}  reg={reg.item():.4e}  "
                      f"beta={beta:.1f}")

        qblk.eval()

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------

    @staticmethod
    def _gather_calib(calib_loader) -> torch.Tensor:
        """Collect and interpolate calibration images → CPU [N, 3, 112, 112]."""
        parts = []
        for batch in calib_loader:
            imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            imgs = imgs.cpu()
            if imgs.shape[-1] != 112 or imgs.shape[-2] != 112:
                imgs = F.interpolate(imgs, size=112)
            parts.append(imgs)
        return torch.cat(parts, dim=0)

    def _build_param_groups(self, ada_params, lsq_params):
        groups = []
        if self.opt_target in {"both", "weights"}:
            groups.append({"params": ada_params, "lr": 1e-3})
        if self.opt_target in {"both", "activations"}:
            groups.append({"params": lsq_params, "lr": 4e-4})
        if not groups:
            raise ValueError(f"No optimizer params for opt_target={self.opt_target}")
        return groups

    def _adaround_reg_loss(self, qblk, ada_params, beta: float,
                           coef: float) -> torch.Tensor:
        if coef == 0.0 or self.opt_target == "activations":
            return torch.zeros((), device=self.device)

        reg = qblk.adaround_reg(beta)
        if self.reg_reduction == "mean":
            denom = sum(p.numel() for p in ada_params)
            reg = reg / max(denom, 1)
        return reg * coef
