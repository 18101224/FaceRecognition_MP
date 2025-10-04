#!/bin/bash
#SBATCH -J hcir_fer
#SBATCH -A m1248_g 
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
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


# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# export MKL_NUM_THREADS=${OMP_NUM_THREADS}
# export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

# NODELIST=$(scontrol show hostnames "$SLURM_NODELIST")
# NODE1=$(echo "$NODELIST" | sed -n '1p')

python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=256 --n_epochs=50 --weight_decay=5e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet8 --num_classes=8 --use_sampler=True --img_size=112 \
--model_type=kprpe12m \
--mean_weight=checkpoint/kbcl_kprpe12m_af8 \
--loss=EKCL --k_grad=True --balanced_cl=True --utilize_target_centers=True --moco_k=256 --kcl_k=5 --beta=0.3 --temperature=0.1 