#!/bin/bash
#SBATCH -J hcir_cifar
#SBATCH -A m1248_g 
#SBATCH -q debug
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=alswo01287@naver.com
#SBATCH --mail-type=ALL

source /pscratch/sd/s/sgkim/hcir/mc/bin/activate
conda activate /pscratch/sd/s/sgkim/hcir/cv 


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

LEARNING_RATES=(0.15 0.1 )

make_cmd () {

  local EXTRA=$1        # loss·weight·스케줄 인자 묶음
  python3 train_imbalanced.py \
        --batch_size=256 --n_epochs=200 --weight_decay=5e-4 \
        --cos=True --momentum=0.9 --world_size=1 --wandb_token=../wandb.txt \
        --model_type=e2_resnet32  --imb_type=exp --imb_factor=0.01 \
        --dataset_path=../data --aug=True --cutout=True --use_wandb=True  --feature_branch=True  --regular_simplex=True --feature_module=deepcomplex_3 --use_tf=True \
         --cosine_scaling=32   --temperature=0.1 --scheduler=warmup $EXTRA
}


for lr in 0.1 0.15; do
  srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK:-1} --cpu-bind=cores --gpus=1 --gpu-bind=map_gpu:3 bash -lc "$(declare -f make_cmd); make_cmd  '--learning_rate=$lr --dataset_name=cifar100 --use_mean=True --loss=BCL_ECE --ece_weight=1.2 '" &  

  srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK:-1} --cpu-bind=cores --gpus=1 --gpu-bind=map_gpu:0 bash -lc "$(declare -f make_cmd); make_cmd  '--learning_rate=$lr --dataset_name=cifar100 --use_mean=True --loss=BCL_ECE --ece_weight=1 '" &  

  srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK:-1} --cpu-bind=cores --gpus=1 --gpu-bind=map_gpu:1 bash -lc "$(declare -f make_cmd); make_cmd  '--learning_rate=$lr --dataset_name=cifar100 --use_mean=True --loss=BCL_ECE --ece_weight=0.3'" &  
                          
  srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK:-1} --cpu-bind=cores --gpus=1 --gpu-bind=map_gpu:2 bash -lc "$(declare -f make_cmd); make_cmd  '--learning_rate=$lr --dataset_name=cifar100 --use_mean=True --loss=BCL_ECE --ece_weight=0.5'" &  

  wait
done

wait

srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK:-1} --cpu-bind=cores --gpus=1 --gpu-bind=map_gpu:3 bash -lc "$(declare -f make_cmd); make_cmd  '--learning_rate=0.1 --dataset_name=cifar10 --loss=BCL '" &  

srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK:-1} --cpu-bind=cores --gpus=1 --gpu-bind=map_gpu:0 bash -lc "$(declare -f make_cmd); make_cmd  '--learning_rate=0.15 --dataset_name=cifar10 --loss=BCL '" &  

srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK:-1} --cpu-bind=cores --gpus=1 --gpu-bind=map_gpu:1 bash -lc "$(declare -f make_cmd); make_cmd  '--learning_rate=0.1 --dataset_name=cifar10 --loss=CE '" &  
                        
srun --exclusive -N1 -n1 -c ${SLURM_CPUS_PER_TASK:-1} --cpu-bind=cores --gpus=1 --gpu-bind=map_gpu:2 bash -lc "$(declare -f make_cmd); make_cmd  '--learning_rate=0.15 --dataset_name=cifar10 --loss=CE '" &  

wait
