#!/bin/bash

#SBATCH -A m1248_g
#SBATCH -J hcir_multi_exp
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 24:00:00
#SBATCH -o logs/latest.log
#SBATCH -e logs/latest_error.log
#SBATCH --mail-user=alswo01287@naver.com
#SBATCH --mail-type=ALL
#SBATCH -C gpu&hbm80g
#SBATCH -q regular

LRS=(0.1 0.1 0.05 0.05)
LOSSES=(CE_BCL CE_ECE_BCL CE_BCL CE_ECE_BCL)
PORTS=(39500 39501 39502 39503)

source /pscratch/sd/s/sgkim/hcir/mc/bin/activate
conda activate /pscratch/sd/s/sgkim/hcir/cv
export URL="https://api.pushover.net/1/messages.json"
export alarm_user="u2258jmd7zfuqh7r3ap4rx9bo6szh9"
export alarm_app="a8ekwdu7tj37drid8o5q6sudd5f6ed"

for IDX in 0 1 2 3
do
  LR=${LRS[$IDX]}
  LOSS=${LOSSES[$IDX]}
  CUDA_VISIBLE_DEVICES=$IDX \
  python train_imbalanced.py \
    --learning_rate=$LR --batch_size=256 --n_epochs=90 --weight_decay=5e-4 \
    --wandb_token=../wandb.txt --server=pm --dataset_name=imagenet_lt --dataset_path=../data/imgnet \
    --cosine_scaling=32 --cosine_constant_margin=0.5 --model_type=resnet50 --loss=$LOSS --aug=True \
    --randaug_n=2 --randaug_m=10 --use_warmup=True --ce_weight=2 --cl_weight=0.6 --ece_weight=1 --cutout=True --ece_scheduling=cosine \
    > logs/latest_$IDX.log 2> logs/latest_error_$IDX.log &
done
wait

