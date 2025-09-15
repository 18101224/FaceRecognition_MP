#!/bin/bash
#SBATCH -J hcir_qcs
#SBATCH -A m1248_g 
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -t 12:00:00
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


# Launch a single 4-GPU experiment with torchrun
python3 adv_training.py --world_size=1 --use_tf=True --num_workers=32 \
--model_type=ir50 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced \
--id_strategy=masking --n_blocks=80 --detach_lowlevel=True --beta=0.3 --partial_update=True \
--learning_rate=1e-5 --batch_size=128 --n_epochs=200 --use_sampler=True \
