#!/bin/bash
#SBATCH -J hcir_fer
#SBATCH -A m1248_g 
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -t 07:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
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


python3 FER_CL.py --num_workers=32 --use_tf=True \
--learning_rate=1e-5 --batch_size=256 --n_epochs=100 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=CAER --dataset_path=../data/CAER-S --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=kprpe12m  \
--loss=CE 