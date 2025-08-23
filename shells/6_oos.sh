#!/bin/bash
# run_all_cifar100.sh

LEARNING_RATES=(0.15 0.1 0.07 0.05)

make_cmd () {
  local DEVICE=$1
  local LR=$2
  local EXTRA=$3        # loss·weight·스케줄 인자 묶음
  CUDA_VISIBLE_DEVICES="$DEVICE" python3 train_imbalanced.py \
        --learning_rate="$LR" --batch_size=256 --n_epochs=200 --weight_decay=5e-4 \
        --cos=True --momentum=0.9 --world_size=1 --wandb_token=../wandb.txt \
        --model_type=resnet32 --dataset_name=cifar100 --imb_type=exp --imb_factor=0.01 \
        --dataset_path=../data --aug=True --cutout=True --use_wandb=True  --feature_branch=True --regular_simplex=True --feature_module=deepcomplex_3 --use_tf=True $EXTRA
}


for LR in "${LEARNING_RATES[@]}"; do

  make_cmd "0" "$LR" "--loss=BCL --ce_weight=2 --cl_weight=0.6   --cosine_scaling=32  \
                          --temperature=0.1 --scheduler=warmup" &  

  make_cmd "1" "$LR" "--loss=BCL --ce_weight=2 --cl_weight=0.6   --cosine_scaling=32  \
                          --temperature=0.1 --scheduler=warmup" &  
                          
  make_cmd "2" "$LR" "--loss=BCL --ce_weight=2 --cl_weight=0.6 --cosine_scaling=32  \
                          --temperature=0.1 --scheduler=warmup" &  


   wait          # ---- 네 개 모두 종료될 때까지 대기 ----
done

