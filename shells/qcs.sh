#!/bin/bash
#SBATCH -J hcir_qcs
#SBATCH -A m1248_g 
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 6:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=alswo01287@naver.com
#SBATCH --mail-type=ALL
#SBATCH -C gpu


source /pscratch/sd/s/sgkim/hcir/mc/bin/activate
conda activate /pscratch/sd/s/sgkim/hcir/cv 


# Launch a single 4-GPU experiment with torchrun
torchrun \
  --nproc_per_node=4 \
  --master_port=${MASTER_PORT:-29501} \
  QCS_hcm.py \
    --dataset_path=../data/RAF-DB_balanced \
    --dataset_name=RAF-DB \
    --guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e \
    --num_workers=32 \
    --world_size=4 \
    --batch_size=64 \
    --n_epochs=200 \
    --learning_rate=2e-5 \
    --use_sampler=True --use_tf=True \
    --model_type=kp_rpe --pin_memory=True --loss=HCM --cl_weight=0.3