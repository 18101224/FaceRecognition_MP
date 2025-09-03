#!/bin/bash
make_cmd(){
  param1=$1 
  CUDA_VISIBLE_DEVICES=2 python3 train_imbalanced.py --world_size=1 \
      --learning_rate=0.15 --batch_size=256 --n_epochs=200 --scheduler=warmup --weight_decay=5e-4 \
      --warmup_epochs=5 --cos=True --use_tf=True --num_workers=16 --use_wandb=True \
      --cosine_scaling=32 --loss=BCL_ECE --ce_weight=1 --cl_weight=1  --model_type=resnet32 --feature_branch=True \
      --dataset_name=cifar10 --dataset_path=../data --imb_factor=0.01 --aug=True --cutout=True $param1
}


make_cmd '--ece_weight=0.3 --surrogate=True' &
make_cmd '--ece_weight=0.5 --surrogate=True' &
make_cmd '--ece_weight=1 --surrogate=True' &
wait
