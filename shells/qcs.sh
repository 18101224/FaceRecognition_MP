#!/bin/bash
#SBATCH -J hcir_qcs
#SBATCH -A m1248_g 
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=alswo01287@naver.com
#SBATCH --mail-type=ALL
#SBATCH -C gpu&hbm80g




for lr in 2e-5 9e-6 1e-5 3e-5; do
srun --exclusive -N1 -n1 --gpus=1 --cpus-per-task=32 \
  python3 QCS_hcm.py \
  --dataset_path=../data/RAF-DB_balanced \
  --dataset_name=RAF-DB \
  --guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e \
  --num_workers=32 \
  --world_size=1 \
  --batch_size=128 \
  --n_epochs=200 \
  --learning_rate="$lr" \
  --use_sampler=True \
  --model_type=ir50 &
done  

wait 