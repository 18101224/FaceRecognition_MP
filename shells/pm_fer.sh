#!/bin/bash
#SBATCH -J hcir_fer
#SBATCH -A m1248_g 
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -t 07:00:00
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=2
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


CUDA_VISIBLE_DEVICES=0 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-5 --batch_size=256 --n_epochs=200 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=kprpe12m --feature_branch=True --use_bn=True \
--loss=KBCL_ETF --kcl_k=10 --moco_k=96 --utilize_target_centers=True  --etf_weight=1 --balanced_cl=True --temperature=0.1 &

CUDA_VISIBLE_DEVICES=1 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-5 --batch_size=256 --n_epochs=200 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=kprpe12m --feature_branch=True --use_bn=True \
--loss=KBCL_ETF --kcl_k=10 --moco_k=96 --utilize_target_centers=True  --etf_weight=1 --balanced_cl=True --temperature=0.1 &

wait 

python3 -c "from utils.pushover import send_message; send_message(' pm_fer finished')" 