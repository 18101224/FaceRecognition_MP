#!/bin/bash
#SBATCH -J hcir_fer
#SBATCH -A m1248_g 
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=alswo01287@naver.com
#SBATCH --mail-type=ALL
#SBATCH -C gpu&hbm80g

source /pscratch/sd/s/sgkim/hcir/mc/bin/activate
conda activate /pscratch/sd/s/sgkim/hcir/cv 


# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# export MKL_NUM_THREADS=${OMP_NUM_THREADS}
# export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

# NODELIST=$(scontrol show hostnames "$SLURM_NODELIST")
# NODE1=$(echo "$NODELIST" | sed -n '1p')


make_cmd () {

  local EXTRA=$1        # loss·weight·스케줄 인자 묶음
  python3 train_imbalanced.py \
        --batch_size=256 --n_epochs=200 --weight_decay=5e-4 \
          --cos=True --momentum=0.9 --world_size=1 \
        --model_type=kp_rpe  --imb_type=exp --imb_factor=0.01 \
        --dataset_path=../data/RAF-DB_aligned --dataset_name=RAF-DB --use_sampler=True --aug=True --cutout=True --use_wandb=True  --feature_branch=True --use_tf=True --cosine_constant_margin=0.5 --num_workers=32 \
           --temperature=0.1 --scheduler=warmup  $EXTRA
}



CUDA_VISIBLE_DEVICES=0 make_cmd "--learning_rate=1e-5 --loss=CE --cl_weight=1 --cosine_scaling=1"  &
CUDA_VISIBLE_DEVICES=1 make_cmd "--learning_rate=1e-5 --loss=CE --cl_weight=1 --cosine_scaling=4"  &
CUDA_VISIBLE_DEVICES=2 make_cmd "--learning_rate=1e-5 --loss=CE --cl_weight=1 --cosine_scaling=16" & 
CUDA_VISIBLE_DEVICES=3 make_cmd "--learning_rate=1e-5 --loss=CE --cl_weight=1 --cosine_scaling=32" &
wait  