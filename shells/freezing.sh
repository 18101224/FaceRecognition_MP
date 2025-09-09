#!/bin/bash
#SBATCH -J hcir_qcs
#SBATCH -A m1248_g 
#SBATCH -q shared
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --mail-user=alswo01287@naver.com
#SBATCH --mail-type=ALL
#SBATCH -C gpu




#residual_3
for lr in 1e-5 2e-5; do 
# full- finetuning 
python3 QCS_hcm.py --learning_rate=$lr --batch_size=128 --n_epochs=200 --world_size=1 --weight_decay=1e-4 \
            --guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e --dataset_path=../data/RAF-DB_balanced --dataset_name=RAF-DB \
            --use_sampler=True --model_type=ir50 --num_workers=32 --use_tf=True --loss=CE --pin_memory=True --learnable_input_dist=True \
            
    for feature_module in residual_1 residual_3 residual_5 ; do
    python3 QCS_hcm.py --learning_rate=$lr --batch_size=128 --n_epochs=200 --world_size=1 --weight_decay=1e-4 \
                --guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e --dataset_path=../data/RAF-DB_balanced --dataset_name=RAF-DB \
                --use_sampler=True --model_type=ir50 --num_workers=32 --use_tf=True --loss=CE --pin_memory=True --learnable_input_dist=True \
                --feature_module=$feature_module
    done

python3 QCS_hcm.py --learning_rate=$lr --batch_size=128 --n_epochs=200 --world_size=1 --weight_decay=1e-4 \
            --guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e --dataset_path=../data/RAF-DB_balanced --dataset_name=RAF-DB \
            --use_sampler=True --model_type=ir50 --num_workers=32 --use_tf=True --loss=CE --pin_memory=True --learnable_input_dist=True --freeze_backbone=True \
            
    for feature_module in residual_1 residual_3 residual_5 ; do
    python3 QCS_hcm.py --learning_rate=$lr --batch_size=128 --n_epochs=200 --world_size=1 --weight_decay=1e-4 \
                --guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e --dataset_path=../data/RAF-DB_balanced --dataset_name=RAF-DB \
                --use_sampler=True --model_type=ir50 --num_workers=32 --use_tf=True --loss=CE --pin_memory=True --learnable_input_dist=True --freeze_backbone=True \
                --feature_module=$feature_module
    done

done