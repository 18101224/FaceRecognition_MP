#!/bin/bash

POSTFIX="best_acc.pth"
POSTFIX3="analysis"

MODELS=(checkpoint/3081306370197 checkpoint/3081306370194 checkpoint/3081306370197 checkpoint/3081306370192)



# for compare
# python3 analyze_classifier.py --dataset_name=cifar10 --model_paths ${MODELS[1]} ${MODELS[2]} --save_path="results/cifar10_bclvsece" \
#  --mode=compare --dataset_path=../data --imb_factor=0.01  --model_type=resnet32 --model_names BCL BCL_ECE 

# for model in ${MODELS[@]}; do
#     python3 analyze_classifier.py --dataset_name=cifar10 --model_paths ${model} --save_path="results/${model}" \
#         --mode=analysis --dataset_path=../data --imb_factor=0.01  --model_type=resnet32 --model_names ECE_cifar10
# done

python3 analyze_classifier.py --dataset_name=cifar10 --model_paths checkpoint/3081504081673 --save_path="results/ECE_cifar10_residual_wo_feature_regular_simplex" \
    --mode=analysis --dataset_path=../data --imb_factor=0.01  --model_type=resnet32 --model_names ECE_cifar10_regular_simplex