#!/bin/bash
#SBATCH -J hcir_qcs
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
#SBATCH -C gpu

source /pscratch/sd/s/sgkim/hcir/mc/bin/activate
conda activate /pscratch/sd/s/sgkim/hcir/cv 


python3 QCS_hcm.py --learning_rate=1e-5 --batch_size=128 --n_epochs=200 --world_size=1 --num_workers=32 \
 --use_tf=True --pin_memory=True --loss=CE --dataset_path=../data/RAF-DB_balanced --dataset_name=RAF-DB --model_type=kp_rpe \
 --use_sampler=True --learnable_input_dist=True --feature_module=residual_3 