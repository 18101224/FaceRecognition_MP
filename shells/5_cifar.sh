#!/bin/bash

PID_FILE="results/running_pids.txt"
: > $PID_FILE  # 파일 초기화

run_block () {
    CUDA=$1
    CMD=$2
    CUDA_VISIBLE_DEVICES=$CUDA bash -c "$CMD" &
    echo $! >> $PID_FILE
}

# CE (GPU 0)
BLOCK_CE="
LEARNING_RATES=(0.5 0.1 0.05 0.07 0.03 0.01 0.007 0.005)
for LR in \${LEARNING_RATES[@]}; do
    python3 train_imbalanced.py \
        --learning_rate=\$LR --batch_size=256 --n_epochs=600 --weight_decay=1e-4 \
        --world_size=1 --wandb_token=../wandb.txt --server=5 \
        --dataset_name=cifar100 --imb_type=exp --imb_factor=100 --dataset_path=../data \
        --s=32 --cosine_constant_margin=0.5 --model_type=cosine --num_hidden_layers=0 --loss=CE \
        --use_warmup=True 
done
python3 -c 'from utils import send_message; send_message(\"ce done\")'
"

# BCL (GPU 1)
BLOCK_BCL="
LEARNING_RATES=(0.5 0.1 0.05 0.07 0.03 0.01 0.007 0.005)
for LR in \${LEARNING_RATES[@]}; do
    python3 train_imbalanced.py \
        --learning_rate=\$LR --batch_size=256 --n_epochs=600 --weight_decay=1e-4 \
        --world_size=1 --wandb_token=../wandb.txt --server=5 \
        --dataset_name=cifar100 --imb_type=exp --imb_factor=10 --dataset_path=../data \
        --s=32 --cosine_constant_margin=0.5 --model_type=cosine --num_hidden_layers=0 --loss=BCL --cl_views=sim-sim \
         --randaug_n=2 --randaug_m=10 --use_warmup=True --ce_weight=2 --cl_weight=0.6 
done
python3 -c 'from utils import send_message; send_message(\"bcl done\")'
"

# BCL with ECE (GPU 2)
BLOCK_BCL_ECE="
LEARNING_RATES=(0.5 0.1 0.05 0.07 0.03 0.01 0.007 0.005)
for LR in \${LEARNING_RATES[@]}; do
    python3 train_imbalanced.py \
        --learning_rate=\$LR --batch_size=256 --n_epochs=600 --weight_decay=1e-4 \
        --world_size=1 --wandb_token=../wandb.txt --server=5 \
        --dataset_name=cifar100 --imb_type=exp --imb_factor=10 --dataset_path=../data \
        --s=32 --cosine_constant_margin=0.5 --model_type=cosine --num_hidden_layers=0 --loss=CE_ECE_BCL --cl_views=sim-sim \
         --randaug_n=2 --randaug_m=10 --use_warmup=True --ce_weight=2 --cl_weight=0.6 --ece_weight=2
done
python3 -c 'from utils import send_message; send_message(\"ece done\")'
"

run_block 1 "$BLOCK_CE"
run_block 2 "$BLOCK_BCL"
run_block 3 "$BLOCK_BCL_ECE"

wait