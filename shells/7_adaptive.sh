#!/bin/bash
make_cmd () {

  local EXTRA=$1        # loss·weight·스케줄 인자 묶음
  python3 train_imbalanced.py \
        --batch_size=256 --n_epochs=200 --weight_decay=5e-4 \
          --cos=True --momentum=0.9 --world_size=1 \
         --imb_type=exp --imb_factor=0.01 \
        --dataset_path=../data --aug=True --cutout=True --use_wandb=True  --feature_branch=True --use_tf=True --num_workers=16 \
         --cosine_scaling=32    --temperature=0.1 --scheduler=warmup --cl_weight=1 --ce_weight=1 $EXTRA
}

CUDA_VISIBLE_DEVICES=0 make_cmd '--learning_rate=0.15 --cl_weight=1 --ce_weight=1 --dataset_name=cifar100 --loss=BCL_ECE --surrogate=True --use_mean=True --model_type=resnet32_128d --mean_weight=1 --std_weight=0.5 --ece_weight=0.3' &