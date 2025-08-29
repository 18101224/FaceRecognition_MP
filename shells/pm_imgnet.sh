#!/bin/bash
#SBATCH -J hcir_cifar
#SBATCH -A m1248_g 
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 32:00:00
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=2
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

NODELIST=$(scontrol show hostnames "$SLURM_NODELIST")
NODE1=$(echo "$NODELIST" | sed -n '1p')
NODE2=$(echo "$NODELIST" | sed -n '2p')


make_cmd () {

  local EXTRA=$1        # loss·weight·스케줄 인자 묶음
  local PORT=$2
  torchrun --nproc_per_node=2 --master_port=$PORT train_imbalanced.py \
        --batch_size=256 --n_epochs=90 --weight_decay=5e-4 \
          --cos=True --momentum=0.9 --world_size=2 \
        --model_type=resnet50  --imb_type=exp --imb_factor=0.01 \
        --dataset_path=../data/imgnet --aug=True --cutout=True --use_wandb=True  --feature_branch=True --use_tf=True \
         --cosine_scaling=32    --temperature=0.1 --scheduler=cosine --num_workers=16 $EXTRA
}


for lr in 0.1 0.15 ; do
srun --exclusive -N1 -n1 -w $NODE1 -c ${SLURM_CPUS_PER_TASK:-1} --cpu-bind=cores --gpus=2 --gpu-bind=map_gpu:2,3 bash -lc "$(declare -f make_cmd); make_cmd  '--learning_rate=$lr --dataset_name=imagenet_lt  --loss=BCL --ce_weight=1 --cl_weight=0.35  '" &  
srun --exclusive -N1 -n1 -w $NODE1 -c ${SLURM_CPUS_PER_TASK:-1} --cpu-bind=cores --gpus=2 --gpu-bind=map_gpu:0,1 bash -lc "$(declare -f make_cmd); make_cmd  '--learning_rate=$lr --dataset_name=imagenet_lt  --loss=CE --ce_weight=1   '" &  
wait
