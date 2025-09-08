#!/bin/bash
#SBATCH -J hcir_qcs
#SBATCH -A m1248_g 
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=alswo01287@naver.com
#SBATCH --mail-type=ALL
#SBATCH -C gpu&hbm80g


source /pscratch/sd/s/sgkim/hcir/mc/bin/activate
conda activate /pscratch/sd/s/sgkim/hcir/cv 


# Launch as one job step with 4 tasks; bind each task to a distinct GPU
srun --exclusive --ntasks=4 --gpus-per-task=1 --cpus-per-task=32 --gpu-bind=map_gpu:0,1,2,3 \
  bash -c 'lr=${@:$(($SLURM_LOCALID+1)):1}; \
  echo "Task $SLURM_LOCALID using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, lr=$lr"; \
  python3 QCS_hcm.py \
    --dataset_path=../data/RAF-DB_balanced \
    --dataset_name=RAF-DB \
    --guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e \
    --num_workers=32 \
    --world_size=1 \
    --batch_size=64 \
    --n_epochs=200 \
    --learning_rate="$lr" \
    --use_sampler=True --use_tf=True \
    --model_type=ir50 --pin_memory=True --use_hcm=True ' _ 2e-5 9e-6 1e-5 3e-5