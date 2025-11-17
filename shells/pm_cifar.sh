#!/bin/bash
#SBATCH -J hcir_raf
#SBATCH -A m1248_g 
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 17:00:00
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

torchrun --nproc_per_node=4 FER_CL.py -world_size=4 --num_workers=32 \
--learning_rate=1e-6 --batch_size=256 --n_epochs=30 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=kprpe12m --resume_path=checkpoint/pmf9467999-0e31-47e1-a172-a41704c4b465 \
--loss=BEAC --beta=2 --utilize_target_centers=True