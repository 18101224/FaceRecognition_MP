#!/bin/bash

#SBATCH -A m1248_g
#SBATCH -J hcir_oos_target
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -o logs/hcir.log
#SBATCH -e logs/hcir_error.log
#SBATCH --mail-user=alswo01287@naver.com
#SBATCH --mail-type=ALL
#SBATCH -C gpu&hbm80g
#SBATCH -q debug


export URL="https://api.pushover.net/1/messages.json"
export alarm_user="u2258jmd7zfuqh7r3ap4rx9bo6szh9"
export alarm_app="a8ekwdu7tj37drid8o5q6sudd5f6ed"

# Learning rate 리스트
export LR_0=0.000001
export LR_1=0.000005


source /pscratch/sd/s/sgkim/hcir/mc/bin/activate
conda activate /pscratch/sd/s/sgkim/hcir/cv 


srun --nodes=1 --ntasks=2 --gpus-per-node=4 --gpus-per-task=2 --cpu-bind=none --gpu-bind=none \
bash -c '
case $SLURM_LOCALID in
      0) PORT=37811 ;;
      1) PORT=37812 ;;
    esac

# Experiment 1: with instance_ada_loss and confidence_constant
torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT OOS_target_train.py --world_size=2 \
  --learning_rate=0.0001 --batch_size=128 --pretrained=True --n_epochs=80 --architecture=resnext50 \
  --dataset_path=../data/clothing1m --dataset_name=clothing1m --num_classes=14 \
  --instance_ada_loss=True --cos_constant_margin=0.5 --confidence_constant=1 --num_workers=32 \
  --n_folds=5 --g_net_ckpt=checkpoints/clothing_best --random_seed=566 --oos_tensor=checkpoint/clothing1m_conf_db.pt \
  > logs/clothing1m_with_instance_ada.log 2>&1

# Experiment 2: without instance_ada_loss and without confidence_constant
torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT OOS_target_train.py --world_size=2 \
  --learning_rate=0.0001 --batch_size=128 --pretrained=True --n_epochs=80 --architecture=resnext50 \
  --dataset_path=../data/clothing1m --dataset_name=clothing1m --num_classes=14 \
  --cos_constant_margin=0.5 --num_workers=32 \
  --n_folds=5 --g_net_ckpt=checkpoints/clothing_best --random_seed=566 --oos_tensor=checkpoint/clothing1m_conf_db.pt \
  > logs/clothing1m_without_instance_ada.log 2>&1
'