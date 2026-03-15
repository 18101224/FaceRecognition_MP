from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import os
from pathlib import Path
import sys
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from arguments import get_arguments
from dataset import get_loader
from losses import get_margin_loss
from models import get_classifier, get_model
from opt import build_scheduler, get_last_lr, get_optimizer, scheduler_step
from utils.logging import get_id



class Trainer:
    def __init__(self, args):
        self.args = args
        self.accelerator = self._build_accelerator()
        self.device = self._resolve_device()
        self._sync_runtime_from_backend()
        self._apply_resume_runtime_overrides()
        self.scaler = self._build_grad_scaler()
        self.wandb_log_interval = 100

        self.run_id = get_id()
        self.global_step = 0
        self.start_epoch = 0
        self.best_loss = float("inf")
        self.logs = defaultdict(list)
        self.aligner = self._build_aligner()
        self.model = self._build_model()
        self.loader, self.num_classes = self._build_dataloader()
        self.loss_fn = self._build_loss()
        self._setup_backbone()
        self.classifier = self._build_classifier()
        self._setup_classifier_optimization()

        if self.args.resume_path:
            self.load_resume_path(self.args.resume_path)

        self.run_dir = self._resolve_run_dir()
        self.wandb_run = self._setup_wandb()

    # Research-facing methods: these are the first places you usually edit.

    def fit(self):
        for epoch in range(self.start_epoch, self.args.n_epochs):
            self.epoch = epoch
            previous_best = self._distributed_reduce_scalar(self.best_loss, reduce_op="min")
            self.best_loss = previous_best
            epoch_loss = self.run_train_epoch()
            self.save_resume_path(tag="last")
            if epoch_loss <= previous_best:
                self.save_resume_path(tag="best")
        self._finish_wandb()

    def run_train_epoch(self):
        self.model.train()
        self.classifier.train()

        sampler = getattr(self.loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.epoch)

        progress = tqdm(self.loader, disable=not self.is_main_process, desc=f"Epoch {self.epoch}")
        epoch_loss_sum = 0.0
        epoch_steps = 0
        epoch_start_time = time.perf_counter()
        epoch_samples_local = 0
        ema_step_time = None
        world_size = max(int(self.args.world_size), 1)
        gpu_mem_postfix = "-"

        for batch in progress:
            step_start_time = time.perf_counter()
            images, labels, keypoints = self._split_batch(batch)
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if keypoints is not None:
                keypoints = keypoints.to(self.device, non_blocking=True)
            else:
                _,_,keypoints,_,_,_ = self.aligner(images)

            self.opt.zero_grad(set_to_none=True)
            self.c_opt.zero_grad(set_to_none=True)

            loss = self.run_train_forward(images, labels, keypoints=keypoints)
            if self.accelerator is not None:
                self.accelerator.backward(loss)
            elif self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            optimizer_step_applied = True
            if self.accelerator is not None:
                self.opt.step()
                self.c_opt.step()
            elif self.scaler is not None:
                previous_scale = self.scaler.get_scale()
                self.scaler.step(self.opt)
                self.scaler.step(self.c_opt)
                self.scaler.update()
                optimizer_step_applied = self.scaler.get_scale() >= previous_scale
            else:
                self.opt.step()
                self.c_opt.step()

            if optimizer_step_applied:
                scheduler_step(self.scheduler, self.global_step)
                scheduler_step(self.c_scheduler, self.global_step)

            loss_value = float(loss.detach().item())
            epoch_loss_sum += loss_value
            epoch_steps += 1
            if epoch_steps == 1 or (epoch_steps % 100 == 0):
                gpu_mem_postfix = self._format_gpu_memory_usage_postfix()
            step_samples_local = int(images.shape[0])
            epoch_samples_local += step_samples_local
            step_time = max(time.perf_counter() - step_start_time, 1e-8)
            ema_step_time = step_time if ema_step_time is None else (0.9 * ema_step_time + 0.1 * step_time)
            throughput_local = float(step_samples_local) / step_time
            throughput_global = float(step_samples_local * world_size) / step_time
            throughput_global_ema = float(step_samples_local * world_size) / ema_step_time

            if optimizer_step_applied:
                self.global_step += 1
            backbone_lr = get_last_lr(self.opt)
            classifier_lr = get_last_lr(self.c_opt)

            if optimizer_step_applied and self.global_step % self.wandb_log_interval == 0:
                self._log_wandb(
                    {
                        "trainer/global_step": self.global_step,
                        "trainer/epoch": self.epoch,
                        "train/loss_step": loss_value,
                        "train/backbone_lr_step": backbone_lr,
                        "train/classifier_lr_step": classifier_lr,
                        "train/epoch_progress": float(epoch_steps) / float(max(len(self.loader), 1)),
                        "train/step_time_sec": step_time,
                        "train/throughput_img_s_local": throughput_local,
                        "train/throughput_img_s_global": throughput_global,
                        "train/throughput_img_s_global_ema": throughput_global_ema,
                    }
                )

            if self.is_main_process:
                progress.set_postfix(
                    loss=f"{loss_value:.4f}",
                    lr=f"{backbone_lr:.6f}",
                    ips=f"{throughput_global_ema:.1f}",
                    mem=gpu_mem_postfix,
                    step=self.global_step,
                )

        global_loss_sum = self._distributed_reduce_scalar(epoch_loss_sum, reduce_op="sum")
        global_epoch_steps = self._distributed_reduce_scalar(float(epoch_steps), reduce_op="sum")
        mean_loss = global_loss_sum / max(global_epoch_steps, 1.0)
        backbone_lr = get_last_lr(self.opt)
        classifier_lr = get_last_lr(self.c_opt)
        epoch_time = max(time.perf_counter() - epoch_start_time, 1e-8)
        epoch_samples_global = int(epoch_samples_local * world_size)
        epoch_throughput_global = float(epoch_samples_global) / epoch_time
        self.logs["train/loss"].append(mean_loss)
        self.logs["train/backbone_lr"].append(backbone_lr)
        self.logs["train/classifier_lr"].append(classifier_lr)
        self.best_loss = min(self._distributed_reduce_scalar(self.best_loss, reduce_op="min"), mean_loss)
        if self.is_main_process:
            print(
                f"[Epoch {self.epoch}] time={epoch_time:.2f}s "
                f"throughput_global={epoch_throughput_global:.2f} img/s "
                f"samples_global={epoch_samples_global}"
            )
        self._log_wandb(
            {
                "trainer/global_step": self.global_step,
                "trainer/epoch": self.epoch,
                "train/loss_epoch": mean_loss,
                "train/backbone_lr_epoch": backbone_lr,
                "train/classifier_lr_epoch": classifier_lr,
                "train/best_loss": self.best_loss,
                "train/epoch_time_sec": epoch_time,
                "train/throughput_img_s_epoch_global": epoch_throughput_global,
                "train/samples_epoch_global": epoch_samples_global,
            }
        )
        return mean_loss

    def run_train_forward(self, images, labels, keypoints=None):
        with self._autocast_context():
            if keypoints is None:
                embeddings = self.model(images)
            else:
                embeddings = self.model(images, keypoints)
            loss = self.classifier(embeddings, labels)
        return loss

    def _build_model(self):
        return get_model(self.args)

    def _build_dataloader(self):
        train_transform = self.model.make_train_transform()
        loader, num_classes, steps_per_epoch = get_loader(
            self.args,
            train_transform=train_transform,
            use_distributed_sampler=(not self.args.use_accelerator and self.args.world_size > 1),
        )
        self.args.steps_per_epoch = steps_per_epoch
        return loader, num_classes


    def _build_loss(self):
        return get_margin_loss(m=self.args.m, h=self.args.h)

    def _build_aligner(self):
        if self.args.aligner_ckpt is None:
            raise ValueError("`--aligner_ckpt` is required for KP-RPE training.")
        from aligners import get_aligner

        return get_aligner(self.args.aligner_ckpt).to(self.device)

    def _build_classifier(self):
        return get_classifier(
            classifier=self.args.classifier,
            sample_rate=self.args.cf_sample_rate,
            margin_loss_fn=self.loss_fn,
            output_dim=self.args.embedding_dim,
            num_classes=self.num_classes,
            rank=self.args.rank,
            world_size=self.args.world_size,
        ).to(self.device)

    def _setup_backbone(self) -> None:
        if self.accelerator is not None:
            self.opt, _ = get_optimizer(self.args, self.model, build_scheduler=False)
            self.model, self.opt, self.loader = self.accelerator.prepare(self.model, self.opt, self.loader)
            self.device = self.accelerator.device
            self.args.steps_per_epoch = len(self.loader)
            self._warmup_accelerated_backbone()
            if self._should_compile_backbone():
                self.model = torch.compile(self.model, mode="reduce-overhead")
            self.scheduler = build_scheduler(self.args, self.opt)
            return

        self.model = self.model.to(self.device)
        if self.args.world_size > 1:
            self.model = DDP(self.model, device_ids=[self.args.local_rank], output_device=self.args.local_rank)
        self.opt, self.scheduler = get_optimizer(self.args, self.model)

    def _setup_classifier_optimization(self) -> None:
        self.classifier = self.classifier.to(self.device)
        if self.accelerator is not None:
            self.c_opt, _ = get_optimizer(self.args, self.classifier, build_scheduler=False)
            if self._classifier_requires_ddp_wrap():
                self.classifier, self.c_opt = self.accelerator.prepare(self.classifier, self.c_opt)
            else:
                self.c_opt = self.accelerator.prepare_optimizer(self.c_opt)
        else:
            if self.args.world_size > 1 and self._classifier_requires_ddp_wrap():
                self.classifier = DDP(
                    self.classifier,
                    device_ids=[self.args.local_rank],
                    output_device=self.args.local_rank,
                )
            self.c_opt, _ = get_optimizer(self.args, self.classifier, build_scheduler=False)
        self.c_scheduler = build_scheduler(self.args, self.c_opt)

    # Checkpointing: useful to touch sometimes, but less central than the experiment loop above.

    def save_resume_path(self, tag: str = "last") -> None:
        checkpoint_dir = self.run_dir / tag
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()

        classifier_state = self._unwrap_classifier().state_dict()
        self._save_local_rank(
            classifier_state,
            checkpoint_dir / f"classifier_rank{self.args.rank}.pt",
        )

        rank_state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "logs": dict(self.logs),
            "run_id": self.run_id,
            "num_classes": self.num_classes,
            "backbone_optimizer_state_dict": self.opt.state_dict(),
            "classifier_optimizer_state_dict": self.c_opt.state_dict(),
            "backbone_scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "classifier_scheduler_state_dict": self.c_scheduler.state_dict() if self.c_scheduler is not None else None,
            "args": vars(self.args).copy(),
        }
        self._save_local_rank(rank_state, checkpoint_dir / f"train_state.r{self.args.rank}.pt")

        self._save_main(self._backbone_state_dict_for_save(), checkpoint_dir / "model.pt")
        self._barrier()

    def load_resume_path(self, resume_path: str | Path) -> None:
        checkpoint_dir = self._checkpoint_dir_from_resume(resume_path)
        if self.args.ckpt_path is None:
            self.args.ckpt_path = str(self._checkpoint_root_from_dir(checkpoint_dir))

        model_state = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=False)
        self._unwrap_model().load_state_dict(model_state, strict=True)

        self._unwrap_classifier().load_state_dict_from_path(str(checkpoint_dir / "classifier.pt"))

        rank_state = torch.load(self._rank_state_path(checkpoint_dir), map_location="cpu", weights_only=False)
        resume_args = rank_state.get("args", {})
        if isinstance(resume_args, dict) and resume_args.get("rpe_impl") is not None:
            resume_rpe_impl = str(resume_args["rpe_impl"]).strip().lower()
            current_rpe_impl = str(getattr(self.args, "rpe_impl", "extension")).strip().lower()
            if current_rpe_impl != resume_rpe_impl:
                raise ValueError(
                    f"rpe_impl mismatch after resume load: runtime={current_rpe_impl}, checkpoint={resume_rpe_impl}"
                )
        self.opt.load_state_dict(rank_state["backbone_optimizer_state_dict"])
        self.c_opt.load_state_dict(rank_state["classifier_optimizer_state_dict"])

        if self.scheduler is not None and rank_state["backbone_scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(rank_state["backbone_scheduler_state_dict"])
        if self.c_scheduler is not None and rank_state["classifier_scheduler_state_dict"] is not None:
            self.c_scheduler.load_state_dict(rank_state["classifier_scheduler_state_dict"])

        self.logs = defaultdict(list, rank_state.get("logs", {}))
        self.run_id = rank_state.get("run_id", self.run_id)
        self.global_step = int(rank_state.get("global_step", 0))
        self.start_epoch = int(rank_state.get("epoch", -1)) + 1
        self.best_loss = float(rank_state.get("best_loss", self.best_loss))
        self.best_loss = self._distributed_reduce_scalar(self.best_loss, reduce_op="min")

    def _setup_wandb(self):
        if not self.is_main_process:
            return None
        import wandb

        os.environ["WANDB_RUN_ID"] = self.run_id
        os.environ["WANDB_NAME"] = self.run_id
        if self.args.resume_path:
            os.environ["WANDB_RESUME"] = "must"
        elif "WANDB_RESUME" in os.environ:
            del os.environ["WANDB_RESUME"]

        run = wandb.init(
            project=f"face-recognition{self.args.dataset_name}",
            dir=str(self.run_dir),
            name=self.run_id,
            id=self.run_id,
            resume="must" if self.args.resume_path else None,
            config=self._wandb_config(),
        )
        self.wandb_run = run
        wandb.define_metric("trainer/global_step")
        wandb.define_metric("trainer/epoch")
        wandb.define_metric("train/*", step_metric="trainer/global_step")
        self._log_wandb(
            {
                "trainer/global_step": self.global_step,
                "trainer/epoch": self.start_epoch,
                "trainer/resumed": float(bool(self.args.resume_path)),
            }
        )
        return self.wandb_run

    def _wandb_config(self):
        config = vars(self.args).copy()
        config["run_id"] = self.run_id
        config["num_classes"] = self.num_classes
        config["run_dir"] = str(self.run_dir)
        return config

    def _log_wandb(self, metrics):
        if self.wandb_run is None:
            return
        self.wandb_run.log(metrics)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()

    # Low-level runtime utilities: mostly environment and file handling.

    @property
    def is_main_process(self) -> bool:
        if self.accelerator is not None:
            return self.accelerator.is_main_process
        return self.args.rank == 0

    def _build_accelerator(self):
        if not self.args.use_accelerator:
            return None
        from accelerate import Accelerator, FullyShardedDataParallelPlugin

        mixed_precision = self.args.mixed_precision if self.args.mixed_precision in {"no", "fp16", "bf16"} else "no"
        return Accelerator(
            mixed_precision=mixed_precision,
            fsdp_plugin=FullyShardedDataParallelPlugin(
                use_orig_params=True,
            ),
        )

    def _warmup_accelerated_backbone(self) -> None:
        backbone = self._unwrap_model()
        input_size = tuple(getattr(backbone.config, "input_size", (3, 112, 112)))
        num_keypoints = int(getattr(backbone.config.rpe_config, "num_keypoints", 5))
        dummy_images = torch.zeros(
            1,
            *input_size,
            device=self.device,
            dtype=torch.float32,
        )
        dummy_keypoints = torch.zeros(
            1,
            num_keypoints,
            2,
            device=self.device,
            dtype=torch.float32,
        )
        self.model.eval()
        with torch.no_grad():
            with self.accelerator.autocast():
                _ = self.model(dummy_images, dummy_keypoints)
        self.model.train()

    def _should_compile_backbone(self) -> bool:
        if not str(self.args.architecture).startswith("kprpe"):
            return True
        # KP-RPE is compile-safe only with the Triton/torch.library RPE path.
        return str(getattr(self.args, "rpe_impl", "extension")).strip().lower() == "triton"

    def _resolve_device(self) -> torch.device:
        if self.accelerator is not None:
            return self.accelerator.device
        if torch.cuda.is_available():
            return torch.device("cuda", self.args.local_rank)
        return torch.device("cpu")

    def _build_grad_scaler(self):
        if self.accelerator is not None:
            return None
        if self.device.type != "cuda":
            return None
        if self.args.mixed_precision != "fp16":
            return None
        return torch.cuda.amp.GradScaler()

    def _sync_runtime_from_backend(self) -> None:
        if self.accelerator is not None:
            self.args.rank = self.accelerator.process_index
            self.args.local_rank = self.accelerator.local_process_index
            self.args.world_size = self.accelerator.num_processes

    def _apply_resume_runtime_overrides(self) -> None:
        if not self.args.resume_path:
            return

        checkpoint_dir = self._checkpoint_dir_from_resume(self.args.resume_path)
        rank_state_path = self._rank_state_path(checkpoint_dir)
        rank_state = torch.load(rank_state_path, map_location="cpu", weights_only=False)
        checkpoint_args = rank_state.get("args", {})
        if not isinstance(checkpoint_args, dict):
            raise ValueError(f"Invalid checkpoint args in {rank_state_path}")

        checkpoint_rpe_impl = checkpoint_args.get("rpe_impl")
        if checkpoint_rpe_impl is None:
            return

        checkpoint_rpe_impl = str(checkpoint_rpe_impl).strip().lower()
        current_rpe_impl = str(getattr(self.args, "rpe_impl", "extension")).strip().lower()
        explicit_rpe_impl = "--rpe_impl" in sys.argv
        if explicit_rpe_impl and current_rpe_impl != checkpoint_rpe_impl:
            raise ValueError(
                f"rpe_impl mismatch on resume: cli={current_rpe_impl}, checkpoint={checkpoint_rpe_impl}. "
                "Use the same backend to avoid train/eval graph mismatch."
            )
        self.args.rpe_impl = checkpoint_rpe_impl

    def _resolve_run_dir(self) -> Path:
        root = Path(self.args.ckpt_path) if self.args.ckpt_path else Path("checkpoint")
        run_dir = root / self.run_id
        if self.is_main_process:
            run_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()
        return run_dir

    def _barrier(self) -> None:
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        elif self.args.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _unwrap_model(self):
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        if self.accelerator is not None:
            return self.accelerator.unwrap_model(model)
        if isinstance(model, DDP):
            return model.module
        return model

    def _backbone_state_dict_for_save(self):
        if self.accelerator is not None:
            return self.accelerator.get_state_dict(self.model)
        return self._unwrap_model().state_dict()

    def _unwrap_classifier(self):
        classifier = self.classifier
        if hasattr(classifier, "_orig_mod"):
            classifier = classifier._orig_mod
        if self.accelerator is not None:
            classifier = self.accelerator.unwrap_model(classifier)
        if isinstance(classifier, DDP):
            return classifier.module
        return classifier

    def _classifier_requires_ddp_wrap(self) -> bool:
        classifier = self._unwrap_classifier()
        return bool(getattr(classifier, "apply_ddp", False))

    def _autocast_context(self):
        if self.accelerator is not None:
            return self.accelerator.autocast()
        if self.device.type == "cuda" and self.args.mixed_precision in {"bf16", "fp16"}:
            dtype = torch.bfloat16 if self.args.mixed_precision == "bf16" else torch.float16
            return torch.autocast(device_type="cuda", dtype=dtype)
        return nullcontext()

    def _split_batch(self, batch):
        images, labels = batch[:2]
        keypoints = None
        if len(batch) > 2 and torch.is_tensor(batch[2]):
            keypoints = batch[2]
        return images, labels, keypoints

    def _format_gpu_memory_usage_postfix(self) -> str:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return "cpu"

        device_index = self.device.index if self.device.index is not None else torch.cuda.current_device()
        allocated_gb = float(torch.cuda.memory_allocated(device_index)) / (1024.0 ** 3)
        reserved_gb = float(torch.cuda.memory_reserved(device_index)) / (1024.0 ** 3)
        stats = torch.tensor([allocated_gb, reserved_gb], device=self.device, dtype=torch.float32)

        if self.args.world_size > 1 and torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered_stats = [torch.zeros_like(stats) for _ in range(self.args.world_size)]
            torch.distributed.all_gather(gathered_stats, stats)
            if not self.is_main_process:
                return "-"
            return " ".join(
                f"r{rank}:{float(rank_stats[0].item()):.1f}/{float(rank_stats[1].item()):.1f}G"
                for rank, rank_stats in enumerate(gathered_stats)
            )

        return f"r0:{allocated_gb:.1f}/{reserved_gb:.1f}G"

    def _distributed_reduce_scalar(self, value: float, reduce_op: str = "sum") -> float:
        if not (
            self.args.world_size > 1
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return float(value)

        tensor = torch.tensor([float(value)], device=self.device, dtype=torch.float64)
        if reduce_op == "sum":
            op = torch.distributed.ReduceOp.SUM
        elif reduce_op == "min":
            op = torch.distributed.ReduceOp.MIN
        elif reduce_op == "max":
            op = torch.distributed.ReduceOp.MAX
        else:
            raise ValueError(f"Unsupported reduce_op: {reduce_op}")
        torch.distributed.all_reduce(tensor, op=op)
        return float(tensor.item())

    def _save_main(self, obj, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.is_main_process:
            return
        if self.accelerator is not None:
            self.accelerator.save(obj, str(path))
        else:
            torch.save(obj, path)

    def _save_local_rank(self, obj, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj, path)

    def _checkpoint_dir_from_resume(self, resume_path: str | Path) -> Path:
        candidate = Path(resume_path)
        checkpoint_dir = candidate if candidate.is_dir() else candidate.parent
        if (checkpoint_dir / "model.pt").exists():
            return checkpoint_dir
        nested_last = checkpoint_dir / "last"
        if nested_last.is_dir() and (nested_last / "model.pt").exists():
            return nested_last
        return checkpoint_dir

    def _rank_state_path(self, checkpoint_dir: Path) -> Path:
        rank_path = checkpoint_dir / f"train_state.r{self.args.rank}.pt"
        if rank_path.exists():
            return rank_path
        fallback = checkpoint_dir / "train_state.r0.pt"
        if fallback.exists():
            return fallback
        raise FileNotFoundError(f"No train state found in {checkpoint_dir}")

    def _checkpoint_root_from_dir(self, checkpoint_dir: Path) -> Path:
        if checkpoint_dir.name in {"last", "best"}:
            return checkpoint_dir.parent.parent
        return checkpoint_dir.parent


def main():
    args = get_arguments()
    trainer = Trainer(args)
    trainer.fit()


if __name__ == "__main__":
    main()
