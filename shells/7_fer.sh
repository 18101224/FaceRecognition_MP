#!/bin/bash
make_cmd () {

  local EXTRA=$1        # loss·weight·스케줄 인자 묶음
  python3 train_imbalanced.py \
        --batch_size=128 --n_epochs=200 --weight_decay=5e-4 \
         --momentum=0.9 --world_size=1 \
         --imb_type=exp --imb_factor=0.01 --cos=True \
        --dataset_path=../data/RAF-DB_balanced --aug=True --cutout=True --use_wandb=True --use_sampler=True --feature_branch=True --use_tf=True --num_workers=16 \
         --cosine_scaling=4 --temperature=0.1 --cl_weight=1 --ce_weight=1 $EXTRA
}

#without cos 
for weight in 0.3 0.7 ; do
CUDA_VISIBLE_DEVICES=2 make_cmd '--learning_rate=9e-6 --dataset_name=RAF-DB --loss=BCL --model_type=ir50 --cl_weight=0.35 --ce_weight=1' 
done
wait