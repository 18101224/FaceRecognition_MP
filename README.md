# FaceRecognition

This repository is a research-oriented face recognition pipeline built around KP-RPE training and evaluation.

The project has two goals:

- make KP-RPE training faster through mixed precision and multi-GPU support
- make the codebase easier to extend by keeping the training, preprocessing, dataset, and evaluation pipelines more explicit and controllable

The code is structured so that changing one part of the pipeline is less likely to require ad-hoc edits across the whole repository. In practice, that means:

- dataset handling is centralized through a registry
- preprocessing is separated from training
- training and evaluation entrypoints are explicit
- accelerator/FSDP and standard DDP paths are both supported

This README reflects the current public-facing policy:

- you are expected to prepare datasets yourself
- this repository only documents how already-prepared datasets should be arranged and used
- upstream checkpoints are not redistributed here
- dataset access and usage permissions remain the user's responsibility

## 1. Upstream Provenance

The `models/` and `aligners/` packages in this repository are derived from the official KPRPE repository:

- https://github.com/mk-minchul/kprpe

They were then adapted here so the overall face recognition pipeline is easier to read, modify, and extend for local research work.

This repository does not redistribute upstream pretrained checkpoints. If you need KP-RPE or aligner checkpoints, obtain them from the official KPRPE repository and the resources linked from it:

- https://github.com/mk-minchul/kprpe

## 2. Environment

Use the Docker image below:

```bash
docker pull 18101224/cuda129-rpe:latest
```

Run a container:

```bash
docker run -it --rm --gpus all --ipc=host \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" \
  -v "/path/to/FaceRecognition:/workspace" \
  -v "/data:/data" \
  -w /workspace \
  18101224/cuda129-rpe:latest \
  bash
```

Assumptions used throughout this README:

- the repository is mounted at `/workspace`
- datasets are mounted under `/data`
- commands are executed from `/workspace`

## 3. Repository Scope

This repository covers:

- dataset loading
- MTCNN-based face alignment preprocessing
- KP-RPE training
- mixed precision and multi-GPU training
- verification evaluation

This repository intentionally does not cover:

- mirrored downloads
- redistribution of dataset archives or benchmark files
- dataset licensing guidance

For any dataset used here, prepare it separately and place it under your own local data root.

## 4. Supported Training Datasets

Registered dataset names:

- `ms1mv3`
- `ms1mv2_subset`
- `webface4m`
- `webface12m`
- `vgg2`, `vgg2_aligned`
- `casia`, `casia_aligned`
- `casia_raw`, `casia_parquet`

Recommended dataset names for the workflows below:

- `casia` for aligned CASIA-WebFace folder trees
- `vgg2` for aligned VGGFace2 folder trees

## 5. Dataset Layout Expectations

### 5.1 CASIA aligned

Training with `--dataset_name casia` expects an aligned image-folder dataset such as:

```text
/data/mj/casia-webface-aligned/
  train/
    <identity_1>/
      image1.jpg
      image2.jpg
    <identity_2>/
      image1.jpg
```

### 5.2 CASIA raw parquet

If your local CASIA copy is in parquet form, use:

- `--dataset_name casia_raw`

Expected root examples:

```text
/data/mj/casia-webface-hf/
  data/
    train-00000-of-00020.parquet
    ...
```

This format is useful as preprocessing input, not as the preferred KP-RPE training format.

### 5.3 VGGFace2 aligned

Training with `--dataset_name vgg2` expects a folder tree such as:

```text
/data/mj/vgg2_aligned/
  train/
    n000001/
      xxx.jpg
    n000002/
      yyy.jpg
```

### 5.4 MS1MV3

Training with `--dataset_name ms1mv3` expects the RecordIO layout:

```text
/data/ms1mv3/
  train.rec
  train.idx
```

## 6. Preprocessing

The preprocessing pipeline applies MTCNN 5-point landmark detection and similarity alignment.

This step is recommended before KP-RPE training on CASIA or VGGFace2.

### 6.1 CASIA raw parquet -> aligned folders

```bash
bash shells/preprocessing.sh \
  casia \
  /data/mj/casia-webface-hf \
  /data/mj/casia-webface-aligned \
  cuda:0 32 4
```

### 6.2 VGGFace2 folders -> aligned folders

```bash
bash shells/preprocessing.sh \
  vgg2 \
  /data/mj/vgg2 \
  /data/mj/vgg2_aligned \
  cuda:0 32 4
```

Direct Python entrypoint:

```bash
python preprocessing.py \
  --dataset_name casia \
  --input_root /data/mj/casia-webface-hf \
  --output_root /data/mj/casia-webface-aligned \
  --device cuda:0 \
  --batch_size 32 \
  --num_workers 4 \
  --log_interval 200
```

Notes:

- output is written as `output_root/train/<identity>/*.jpg`
- existing files are skipped by default
- progress is shown through `tqdm`
- for KP-RPE-related training and evaluation, prepare the required aligner checkpoint from the official KPRPE repository before running these commands

## 7. Verification Evaluation Data Preparation

The pair-verification evaluator expects Hugging Face `datasets.save_to_disk` format under a root such as:

```text
/data/mj/facerec_val/
  lfw/
  agedb_30/
  cfp_fp/
  cplfw/
  calfw/
```

### 7.1 Convert existing `.bin` verification sets

If you already have local `.bin` files under `/data/mj/eval_bins`:

```bash
python tools/prepare_verification_eval.py \
  --bin_root /data/mj/eval_bins \
  --out_root /data/mj/facerec_val \
  --names lfw agedb_30 cfp_fp cplfw calfw
```

Check readiness:

```bash
python tools/check_eval_ready.py --root /data/mj
```

### 7.2 Convert local TinyFace / IJB data if already available

If TinyFace and IJB data already exist locally under `/data/mj`, run:

```bash
bash shells/prepare_eval_pipeline.sh /data/mj cuda:0 256 4
python tools/check_eval_ready.py --root /data/mj
```

This script no longer downloads anything. It only:

- converts existing verification `.bin` files
- preprocesses TinyFace if `/data/mj/TinyFace` exists
- preprocesses IJB-C if `/data/mj/IJB_release` exists
- preprocesses IJB-S aligned images if available

Current evaluator coverage:

- `lfw`
- `agedb_30`
- `cfp_fp`
- `cplfw`
- `calfw`

TinyFace and IJB preprocessing utilities are included, but they are not part of the current pair-verification evaluator.

## 8. Training

The current training defaults are fixed in code:

- optimizer: `AdamW`
- scheduler: cosine
- warmup: `3` epochs

Runtime paths:

- `--use_accelerator false`: regular DDP path
- `--use_accelerator true`: `Accelerate + FSDP`

This repository does not use a DeepSpeed backend.

Before running the training commands below, place the required aligner checkpoint under a local path such as:

```text
/workspace/checkpoint/adaface_vit_base_kprpe_webface12m
```

Obtain that checkpoint from the official KPRPE repository and the resources linked from it:

- https://github.com/mk-minchul/kprpe

### 7.1 CASIA, KP-RPE small, without mixed precision

This command uses 2 GPUs and keeps the global batch size at `512`.

```bash
RUN_ID=casia-kprpe-small-nomp CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --standalone train.py \
  --dataset_name casia \
  --dataset_root /data/mj/casia-webface-aligned \
  --aligner_ckpt /workspace/checkpoint/adaface_vit_base_kprpe_webface12m \
  --architecture kprpe_small \
  --embedding_dim 512 \
  --classifier fc \
  --batch_size 512 \
  --n_epochs 100 \
  --learning_rate 1e-3 \
  --weight_decay 0.05 \
  --h 0.333 \
  --mixed_precision no \
  --use_accelerator false \
  --use_flash_attn false \
  --rpe_impl extension
```

### 7.2 CASIA, KP-RPE small, BF16 with Accelerate + FSDP

This command uses 2 GPUs and matches the effective global batch of the no-MP run above.

```bash
RUN_ID=casia-kprpe-small-bf16 CUDA_VISIBLE_DEVICES=2,3 \
torchrun --nproc_per_node=2 --standalone train.py \
  --dataset_name casia \
  --dataset_root /data/mj/casia-webface-aligned \
  --aligner_ckpt /workspace/checkpoint/adaface_vit_base_kprpe_webface12m \
  --architecture kprpe_small \
  --embedding_dim 512 \
  --classifier fc \
  --batch_size 256 \
  --n_epochs 100 \
  --learning_rate 1e-3 \
  --weight_decay 0.05 \
  --h 0.333 \
  --mixed_precision bf16 \
  --use_accelerator true \
  --use_flash_attn false \
  --rpe_impl extension
```

Checkpoint output:

```text
/workspace/checkpoint/<RUN_ID>/
```

Each run stores:

- `best/`
- `last/`
- `train_state.r*.pt`

### 7.3 Resume training

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --standalone train.py \
  --resume_path /workspace/checkpoint/<RUN_ID>/last \
  --dataset_name casia \
  --dataset_root /data/mj/casia-webface-aligned \
  --aligner_ckpt /workspace/checkpoint/adaface_vit_base_kprpe_webface12m \
  --architecture kprpe_small \
  --embedding_dim 512 \
  --classifier fc \
  --batch_size 512 \
  --n_epochs 150 \
  --learning_rate 1e-3 \
  --weight_decay 0.05 \
  --h 0.333 \
  --mixed_precision no \
  --use_accelerator false \
  --use_flash_attn false \
  --rpe_impl extension
```

Note:

- `--n_epochs` is the total target epoch count, not “additional epochs”

## 9. Evaluation

### 8.1 Evaluate one checkpoint

```bash
bash shells/eval_nompi_only.sh \
  /workspace/checkpoint/<RUN_ID> \
  best \
  /data/mj/facerec_val \
  /workspace/checkpoint/adaface_vit_base_kprpe_webface12m \
  256 \
  4 \
  cuda:0 \
  lfw agedb_30 cfp_fp cplfw calfw
```

This wrapper:

- exports the checkpoint to `model.exported.pt` if needed
- runs verification evaluation on the requested datasets

### 8.2 Compare mixed precision vs no mixed precision

```bash
bash shells/eval_mp_vs_nompi.sh \
  /workspace/checkpoint/<MP_RUN_ID> \
  /workspace/checkpoint/<NO_MP_RUN_ID> \
  best \
  /data/mj/facerec_val \
  /workspace/checkpoint/adaface_vit_base_kprpe_webface12m \
  256 \
  4 \
  true \
  lfw agedb_30 cfp_fp cplfw calfw
```

This wrapper:

- exports each checkpoint to `model.exported.pt` if needed
- evaluates both checkpoints sequentially
- reuses exported models on later runs

### 8.3 Re-train an MP checkpoint and compare immediately

```bash
bash shells/retrain_mp_then_eval.sh \
  mp-bf16-fixed \
  /workspace/checkpoint/<NO_MP_RUN_ID> \
  best
```

This wrapper:

- trains a fresh BF16 + accelerator checkpoint
- saves it to `/workspace/checkpoint/mp-bf16-fixed`
- exports it
- evaluates it against an existing no-MP checkpoint

## 10. Checkpoint Caveat

Old accelerator/FSDP checkpoints saved before the full-state save fix may not be evaluable from `model.pt` alone.

Typical failure signs:

- flattened parameter tensors instead of full parameter shapes
- zero-sized tensors in the saved state dict
- export or eval failures caused by local-shard shape mismatches

If that happens, do not use that checkpoint for final comparison. Re-train or re-save the model with the current code.

New accelerator checkpoints created with the current training code save the backbone through `accelerator.get_state_dict(...)`, which is the correct path for later export and evaluation.

## 11. Manual Utilities

Check verification dataset readiness:

```bash
python tools/check_eval_ready.py --root /data/mj
```

Export a non-accelerator checkpoint manually:

```bash
python tools/export_eval_model.py \
  --checkpoint_dir /workspace/checkpoint/<RUN_ID> \
  --checkpoint_tag best \
  --output_path /workspace/checkpoint/<RUN_ID>/best/model.exported.pt
```

Export an accelerator/FSDP checkpoint manually:

```bash
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --standalone \
  tools/export_eval_model.py \
  --checkpoint_dir /workspace/checkpoint/<RUN_ID> \
  --checkpoint_tag best \
  --output_path /workspace/checkpoint/<RUN_ID>/best/model.exported.pt
```

Run evaluation directly:

```bash
python eval_verification.py \
  --checkpoint_dir /workspace/checkpoint/<RUN_ID> \
  --checkpoint_tag best \
  --model_path /workspace/checkpoint/<RUN_ID>/best/model.exported.pt \
  --aligner_ckpt /workspace/checkpoint/adaface_vit_base_kprpe_webface12m \
  --eval_root /data/mj/facerec_val \
  --datasets lfw agedb_30 cfp_fp cplfw calfw \
  --batch_size 256 \
  --num_workers 4 \
  --mixed_precision no \
  --use_flash_attn false \
  --device cuda:0
```
