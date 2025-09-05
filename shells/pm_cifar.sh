#!/bin/bash
#SBATCH -J hcir_cifar
#SBATCH -A m1248_g 
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 10:00:00
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


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

NODELIST=$(scontrol show hostnames "$SLURM_NODELIST")
NODE1=$(echo "$NODELIST" | sed -n '1p')


make_cmd () {

  local EXTRA=$1        # loss·weight·스케줄 인자 묶음
  python3 train_imbalanced.py \
        --batch_size=256 --n_epochs=200 --weight_decay=5e-4 \
          --cos=True --momentum=0.9 --world_size=1 \
         --imb_type=exp --imb_factor=0.01 \
        --dataset_path=../data --aug=True --cutout=True --use_wandb=True  --feature_branch=True --use_tf=True --num_workers=16 \
         --cosine_scaling=32    --temperature=0.1 --scheduler=warmup --cl_weight=1 --ce_weight=1 $EXTRA
}

# CIFAR 10 부터 시작., 

#cifar10 1:1
# for not surrogate to gpu 0
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --model_type=resnet32_64d --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --model_type=resnet32_64d --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --model_type=resnet32_64d --ece_weight=1' &

# for only std to gpu 1 
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --model_type=resnet32_64d --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --model_type=resnet32_64d --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --model_type=resnet32_64d --ece_weight=1' &

# for mean + std to gpu 2 
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --ece_weight=1' &


# for mean + std to gpu 3
CUDA_VISIBLE_DEVICES=3 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL --model_type=resnet32_64d ' &
CUDA_VISIBLE_DEVICES=3 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar10 --loss=BCL --model_type=resnet32_64d ' &
CUDA_VISIBLE_DEVICES=3 make_cmd '--learning_rate=0.1 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL --model_type=resnet32_64d ' &

wait

#cifar100 1:1
# for not surrogate to gpu 0
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --model_type=resnet32_128d --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --model_type=resnet32_128d --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --model_type=resnet32_128d --ece_weight=1' &


# for only std to gpu 1 
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --model_type=resnet32_128d --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --model_type=resnet32_128d --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --model_type=resnet32_128d --ece_weight=1' &

# for mean + std to gpu 2 
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --ece_weight=1' &


# for mean + std to gpu 3
CUDA_VISIBLE_DEVICES=3 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL --model_type=resnet32_128d ' &
CUDA_VISIBLE_DEVICES=3 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar100 --loss=BCL --model_type=resnet32_128d ' &
CUDA_VISIBLE_DEVICES=3 make_cmd '--learning_rate=0.1 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL --model_type=resnet32_128d ' &

wait

# cifar10 0.6:2
# for not surrogate to gpu 0
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar10 --loss=BCL_ECE --model_type=resnet32_64d --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar10 --loss=BCL_ECE --model_type=resnet32_64d --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar10 --loss=BCL_ECE --model_type=resnet32_64d --ece_weight=1' &

# for only std to gpu 1 
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --model_type=resnet32_64d --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --model_type=resnet32_64d --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --model_type=resnet32_64d --ece_weight=1' &

# for mean + std to gpu 2 
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --ece_weight=1' &


# for mean + std to gpu 3
CUDA_VISIBLE_DEVICES=3 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL --model_type=resnet32_64d ' &
CUDA_VISIBLE_DEVICES=3 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar10 --loss=BCL --model_type=resnet32_64d ' &
CUDA_VISIBLE_DEVICES=3 make_cmd '--learning_rate=0.1 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL --model_type=resnet32_64d ' &

wait

# cifar100 0.6:2
# for not surrogate to gpu 0
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=2 --ce_weight=0.6 --dataset_name=cifar100 --loss=BCL_ECE --model_type=resnet32_128d --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=2 --ce_weight=0.6 --dataset_name=cifar100 --loss=BCL_ECE --model_type=resnet32_128d --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=2 --ce_weight=0.6 --dataset_name=cifar100 --loss=BCL_ECE --model_type=resnet32_128d --ece_weight=1' &

# for only std to gpu 1 
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=2 --ce_weight=0.6 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --model_type=resnet32_128d --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=2 --ce_weight=0.6 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --model_type=resnet32_128d --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=2 --ce_weight=0.6 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --model_type=resnet32_128d --ece_weight=1' &

# for mean + std to gpu 2 
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=2 --ce_weight=0.6 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=2 --ce_weight=0.6 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=2 --ce_weight=0.6 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --ece_weight=1' &


# for mean + std to gpu 3
CUDA_VISIBLE_DEVICES=3 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL --model_type=resnet32_128d ' &
CUDA_VISIBLE_DEVICES=3 make_cmd '--learning_rate=0.15 --cl_weight=0.6 --ce_weight=2 --dataset_name=cifar100 --loss=BCL --model_type=resnet32_128d ' &
CUDA_VISIBLE_DEVICES=3 make_cmd '--learning_rate=0.1 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL --model_type=resnet32_128d ' &

wait

# cifar10 weighted 
# for not surrogate to gpu 0
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --mean_weight=1 --std_weight=0.5 --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --mean_weight=1 --std_weight=0.5 --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --mean_weight=1 --std_weight=0.1 --ece_weight=1' &

# for only std to gpu 1 
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --mean_weight=0.5 --std_weight=1 --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --mean_weight=1 --std_weight=0.1 --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --mean_weight=1 --std_weight=0.5 --ece_weight=1' &

# for mean + std to gpu 2 
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --mean_weight=0.1 --std_weight=1 --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --mean_weight=0.1 --std_weight=1 --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --mean_weight=0.1 --std_weight=1 --ece_weight=1' &

# for mean + std to gpu 2 
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --mean_weight=1 --std_weight=0.1 --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --mean_weight=0.5 --std_weight=1 --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar10 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_64d --mean_weight=0.5 --std_weight=1 --ece_weight=1' &
wait

#cifar100 weighted
# for not surrogate to gpu 0
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=1 --std_weight=0.5 --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=1 --std_weight=0.5 --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=1 --std_weight=0.1 --ece_weight=1' &

# for only std to gpu 1 
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=0.5 --std_weight=1 --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=1 --std_weight=0.1 --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=1 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=1 --std_weight=0.5 --ece_weight=1' &

# for mean + std to gpu 2 
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=0.1 --std_weight=1 --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=0.1 --std_weight=1 --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=0.1 --std_weight=1 --ece_weight=1' &

# for mean + std to gpu 2 
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=1 --std_weight=0.1 --ece_weight=0.3' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=0.5 --std_weight=1 --ece_weight=0.5' &
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=0.5 --std_weight=1 --ece_weight=1' &
wait