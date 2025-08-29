#!/bin/bash


make_cmd () {

  local EXTRA=$1        # loss·weight·스케줄 인자 묶음
  python3 train_imbalanced.py \
        --batch_size=256 --n_epochs=200 --weight_decay=5e-4 \
        --cos=True --momentum=0.9 --world_size=1  \
        --model_type=resnet32  --imb_type=exp --imb_factor=0.01 \
        --dataset_path=../data --aug=True --cutout=True --use_wandb=True  --feature_branch=True  --use_tf=True \
         --cosine_scaling=32 --temperature=0.1 --scheduler=warmup --num_workers=16 $EXTRA
}

gpu0 (){
for dataset in cifar100 cifar10 ; do
  for lr in 0.1 0.15; do

    CUDA_VISIBLE_DEVICES=0 make_cmd  "--learning_rate=$lr --dataset_name=$dataset --use_mean=True --loss=BCL_ECE --ce_weight=2 --cl_weight=0.6 --ece_weight=1 --ece_scheduling=cosine" &

    CUDA_VISIBLE_DEVICES=0 make_cmd  "--learning_rate=$lr --dataset_name=$dataset --use_mean=True --loss=BCL_ECE --ce_weight=2 --cl_weight=0.6 --ece_weight=0.3 --ece_scheduling=cosine" &
                            
    CUDA_VISIBLE_DEVICES=0 make_cmd  "--learning_rate=$lr --dataset_name=$dataset --use_mean=True --loss=BCL_ECE --ce_weight=2 --cl_weight=0.6 --ece_weight=0.5 --ece_scheduling=cosine" &

    wait
  done
  wait
done 
}

gpu1 (){
  for dataset in cifar100 cifar10 ; do
    for lr in 0.1 0.15; do
      CUDA_VISIBLE_DEVICES=2 make_cmd  "--learning_rate=$lr --dataset_name=$dataset  --loss=BCL_ECE --ce_weight=2 --cl_weight=0.6 --ece_weight=0.3  " &

      CUDA_VISIBLE_DEVICES=2 make_cmd  "--learning_rate=$lr --dataset_name=$dataset  --loss=BCL_ECE --ce_weight=2 --cl_weight=0.6 --ece_weight=0.5  " &

      CUDA_VISIBLE_DEVICES=2 make_cmd  "--learning_rate=$lr --dataset_name=$dataset  --loss=BCL_ECE --ce_weight=2 --cl_weight=0.6 --ece_weight=0.7  " &
    wait 
    done
    wait
  done
}

gpu0 &
gpu1 &
wait
