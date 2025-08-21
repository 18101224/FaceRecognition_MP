#!/bin/bash

#SBATCH -A m1248_g
#SBATCH -J hcir_oos_target
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 40:00:00
#SBATCH -o logs/hcir.log
#SBATCH -e logs/hcir_error.log
#SBATCH --mail-user=alswo01287@naver.com
#SBATCH --mail-type=ALL
#SBATCH -C gpu&hbm80g
#SBATCH -q regular


export URL="https://api.pushover.net/1/messages.json"
export alarm_user="u2258jmd7zfuqh7r3ap4rx9bo6szh9"
export alarm_app="a8ekwdu7tj37drid8o5q6sudd5f6ed"

# Learning rate 리스트
export LR_0=0.0000001
export LR_1=0.00000007


source /pscratch/sd/s/sgkim/hcir/mc/bin/activate
conda activate /pscratch/sd/s/sgkim/hcir/cv 



srun --nodes=1 --ntasks=2 --gpus-per-node=4 --gpus-per-task=2 --cpu-bind=none --gpu-bind=none \
  bash -c '
    case $SLURM_LOCALID in
      0) LR=$LR_0; PORT=37811 ;;
      1) LR=$LR_1; PORT=37812 ;;
    esac
    torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:$PORT OOS_target_train.py --world_size=2 \
        --n_folds=3 --g_net_ckpt=checkpoints/7080801341664 --random_seed=566 --kprpe_ckpt_path=checkpoint/adaface_vit_base_kprpe_webface12m \
        --cos_constant_margin=0.5 --instance_ada_loss=True \
        --learning_rate=$LR --batch_size=256 --n_epochs=30 \
        --dataset_path=../data/AffectNet7 --dataset_name=AffectNet \
        --wandb_token=../wandb.txt --server=pm \
        --confidence_constant=1 --use_tf=True --cos_scaling=16  \
        --oos_tensor=checkpoint/Affect.pt > logs/lr_${LR}.log 2>&1
  '


