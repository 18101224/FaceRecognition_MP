#!/bin/bash
#SBATCH -J hcir_raf
#SBATCH -A m1248_g 
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 12:00:00
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


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS}

# inside pm_cifar.sh (replace the 4 separate srun blocks)

srun -n 4 -c ${SLURM_CPUS_PER_TASK} \
  --gpus-per-task=1 --gpu-bind=map_gpu:0,1,2,3 --cpu-bind=cores \
  --kill-on-bad-exit=1 -l \
  bash -lc 'i=$SLURM_LOCALID; \
    case "$i" in \
      0) bs=64 ;; \
      1) bs=128 ;; \
      2) bs=32 ;; \
      3) bs=256 ;; \
      *) echo "Unsupported SLURM_LOCALID: $i" >&2; exit 1 ;; \
    esac; \
    python3 MoCo.py --learning_rate=1e-5 --batch_size="$bs" --n_epochs=200 \
    --world_size=1 --num_workers="${SLURM_CPUS_PER_TASK}" --use_tf=True --weight_decay=5e-4 \
    --dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --use_sampler=True \
    --mean_weight=checkpoint/kprpe_72K_${bs} --model_type=kprpe12m \
    --loss=KBCL --kcl_k=5 --beta=0.3 --temperature=0.1 --utilze_class_centers=True --moco_k=72'