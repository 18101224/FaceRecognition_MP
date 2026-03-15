# Dataset Preparation

This repository expects datasets to be prepared locally by the user.

It does not prescribe a download source and does not redistribute dataset archives, benchmark files, or pretrained data assets. Make sure you have the right to use any dataset before placing it under your local data root.

## 1. Registered Training Dataset Names

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

## 2. Expected Dataset Layouts

### 2.1 CASIA aligned

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

### 2.2 CASIA raw parquet

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

### 2.3 VGGFace2 aligned

Training with `--dataset_name vgg2` expects a folder tree such as:

```text
/data/mj/vgg2_aligned/
  train/
    n000001/
      xxx.jpg
    n000002/
      yyy.jpg
```

### 2.4 MS1MV3

Training with `--dataset_name ms1mv3` expects the RecordIO layout:

```text
/data/ms1mv3/
  train.rec
  train.idx
```

## 3. Preprocessing

The preprocessing pipeline applies MTCNN 5-point landmark detection and similarity alignment.

This step is recommended before KP-RPE training on CASIA or VGGFace2.

### 3.1 CASIA raw parquet -> aligned folders

```bash
bash shells/preprocessing.sh \
  casia \
  /data/mj/casia-webface-hf \
  /data/mj/casia-webface-aligned \
  cuda:0 32 4
```

### 3.2 VGGFace2 folders -> aligned folders

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

## 4. Verification Evaluation Data Preparation

The pair-verification evaluator expects Hugging Face `datasets.save_to_disk` format under a root such as:

```text
/data/mj/facerec_val/
  lfw/
  agedb_30/
  cfp_fp/
  cplfw/
  calfw/
```

### 4.1 Convert existing `.bin` verification sets

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

### 4.2 Convert local TinyFace / IJB data if already available

If TinyFace and IJB data already exist locally under `/data/mj`, run:

```bash
bash shells/prepare_eval_pipeline.sh /data/mj cuda:0 256 4
python tools/check_eval_ready.py --root /data/mj
```

This script does not download anything. It only:

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
