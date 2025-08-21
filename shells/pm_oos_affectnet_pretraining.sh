#!/bin/bash

#SBATCH -A m1248_g
#SBATCH -J hcir_oos_target
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH -t 48:00:00
#SBATCH -o logs/hcir_%j.log
#SBATCH -e logs/hcir_%j_error.log
#SBATCH --mail-user=alswo01287@naver.com
#SBATCH --mail-type=ALL
#SBATCH -C gpu&hbm80g
#SBATCH -q shared



source /pscratch/sd/s/sgkim/hcir/mc/bin/activate
conda activate /pscratch/sd/s/sgkim/hcir/cv 


CUDA_VISIBLE_DEVICES=0 python3 OOS_pretraining.py --world_size=1 --batch_size=256 --n_epochs=30 --learning_rate=5e-06 --weight_decay=5e-4 \
    --dataset_path=../data/AffectNet7 --dataset_name=AffectNet --random_seed=566 --wandb_token=../wandb.txt --kprpe_ckpt_path=checkpoint/adaface_vit_base_kprpe_webface12m \
    --num_classes=7 --cos_constant_margin=0.5 --n_folds=3 --server=pm &

CUDA_VISIBLE_DEVICES=1 python3 OOS_pretraining.py --world_size=1 --batch_size=256 --n_epochs=30 --learning_rate=3e-06 --weight_decay=5e-4 \
    --dataset_path=../data/AffectNet7 --dataset_name=AffectNet --random_seed=566 --wandb_token=../wandb.txt --kprpe_ckpt_path=checkpoint/adaface_vit_base_kprpe_webface12m \
    --num_classes=7 --cos_constant_margin=0.5 --n_folds=3 --server=pm &

wait
