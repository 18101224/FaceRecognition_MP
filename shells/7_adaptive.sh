#!/bin/bash


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

  CUDA_VISIBLE_DEVICES=0 make_cmd  "--learning_rate=$lr --dataset_name=cifar10 --use_mean=True --loss=BCL_ECE --ece_weight=1 " &

  CUDA_VISIBLE_DEVICES=1 make_cmd  "--learning_rate=$lr --dataset_name=cifar10 --use_mean=True --loss=BCL_ECE --ece_weight=0.3" &
                          
  CUDA_VISIBLE_DEVICES=2 make_cmd  "--learning_rate=$lr --dataset_name=cifar10 --use_mean=True --loss=BCL_ECE --ece_weight=0.5" &

  wait
done
