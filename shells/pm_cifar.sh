#!/bin/bash
#SBATCH -J hcir_raf
#SBATCH -A m1248_g 
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 20:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
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

SRUN_BASE="srun --ntasks=1 -c ${SLURM_CPUS_PER_TASK} --gpus-per-task=1 --cpu-bind=cores"

${SRUN_BASE} --gpu-bind=map_gpu:0 \
python3 MoCo.py --learning_rate=1e-5 --batch_size=64 --n_epochs=200 --world_size=1 --num_workers=${SLURM_CPUS_PER_TASK} --use_tf=True --weight_decay=5e-4 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --use_sampler=True \
--mean_weight=checkpoint/kprpe_256K_means \
--model_type=kp_rpe \
--loss=KBCL --kcl_k=5 --beta=0.3 --temperature=0.1 --utilze_class_centers=True --moco_k=256 &

${SRUN_BASE} --gpu-bind=map_gpu:1 \
python3 MoCo.py --learning_rate=1e-5 --batch_size=128 --n_epochs=200 --world_size=1 --num_workers=${SLURM_CPUS_PER_TASK} --use_tf=True --weight_decay=5e-4 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --use_sampler=True \
--mean_weight=checkpoint/kprpe_256K_means \
--model_type=kp_rpe \
--loss=KBCL --kcl_k=5 --beta=0.3 --temperature=0.1 --utilze_class_centers=True --moco_k=256 &

${SRUN_BASE} --gpu-bind=map_gpu:2 \
python3 MoCo.py --learning_rate=1e-5 --batch_size=32 --n_epochs=200 --world_size=1 --num_workers=${SLURM_CPUS_PER_TASK} --use_tf=True --weight_decay=5e-4 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --use_sampler=True \
--mean_weight=checkpoint/kprpe_256K_means \
--model_type=kp_rpe \
--loss=KBCL --kcl_k=5 --beta=0.3 --temperature=0.1 --utilze_class_centers=True --moco_k=256 &

${SRUN_BASE} --gpu-bind=map_gpu:3 \
python3 MoCo.py --learning_rate=1e-5 --batch_size=256 --n_epochs=200 --world_size=1 --num_workers=${SLURM_CPUS_PER_TASK} --use_tf=True --weight_decay=5e-4 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --use_sampler=True \
--mean_weight=checkpoint/kprpe_256K_means \
--model_type=kp_rpe \
--loss=KBCL --kcl_k=5 --beta=0.3 --temperature=0.1 --utilze_class_centers=True --moco_k=256 &

wait 