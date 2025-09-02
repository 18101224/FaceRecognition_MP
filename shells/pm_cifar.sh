#!/bin/bash
#SBATCH -J hcir_cifar
#SBATCH -A m1248_g 
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=alswo01287@naver.com
#SBATCH --mail-type=ALL
#SBATCH -C gpu&hbm80g

source /pscratch/sd/s/sgkim/hcir/mc/bin/activate
conda activate /pscratch/sd/s/sgkim/hcir/cv 


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

NODELIST=$(scontrol show hostnames "$SLURM_NODELIST")
NODE1=$(echo "$NODELIST" | sed -n '1p')


make_cmd () {

  local EXTRA=$1        # loss·weight·스케줄 인자 묶음
  python3 train_imbalanced.py \
        --batch_size=256 --n_epochs=200 --weight_decay=5e-4 \
          --cos=True --momentum=0.9 --world_size=1 \
        --model_type=resnet32  --imb_type=exp --imb_factor=0.01 \
        --dataset_path=../data --aug=True --cutout=True --use_wandb=True  --feature_branch=True --use_tf=True --num_workers=32 \
         --cosine_scaling=32    --temperature=0.1 --scheduler=warmup $EXTRA
}




make_cmd '--learning_rate=0.15 --loss=BCL_ECE --ece_weight=0.3 --surrogate=True --dataset_name=cifar100 --ce_weight=1 --cl_weight=1' &
make_cmd '--learning_rate=0.15 --loss=BCL_ECE --ece_weight=0.3 --surrogate=True --dataset_name=cifar100 --ce_weight=1 --cl_weight=1 --hard_weight=1 --soft_weight=0.5 --k=10' &
make_cmd '--learning_rate=0.15 --loss=BCL_ECE --ece_weight=0.5 --surrogate=True --dataset_name=cifar100 --ce_weight=1 --cl_weight=1 --hard_weight=1 --soft_weight=0.5 --k=10' &
wait 
