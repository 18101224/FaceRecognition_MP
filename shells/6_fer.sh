
# CUDA_VISIBLE_DEVICES=0,1 /home/gpuadmin/anaconda3/envs/main/bin/torchrun --nproc_per_node=2 QCS_hcm.py --world_size=2  \
#   --dataset_path=../data/RAF-DB_balanced \
#   --dataset_name=RAF-DB \
#   --guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e \
#   --num_workers=32 \
#   --batch_size=128 \
#   --n_epochs=200 \
#   --learning_rate=1e-5 \
#   --use_sampler=True \
#   --model_type=kp_rpe --loss=CE --cl_weight=0.3 


# head_num_layers1=(1 3 5 7 9 11)
# head_num_layers2=(8 10 12 14 16 18)

# for head_num1 in "${head_num_layers1[@]}";do
# echo $head_num1
# CUDA_VISIBLE_DEVICES=0 python3 QCS_hcm.py --learning_rate=1e-5 --batch_size=128 --n_epochs=200 --world_size=1 --weight_decay=1e-4 \
#             --guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e --dataset_path=../data/RAF-DB_balanced --dataset_name=RAF-DB \
#             --use_sampler=True --model_type=ir50 --num_workers=32 --use_tf=True --loss=CE --pin_memory=True --learnable_input_dist=True --freeze_backbone=True \
#             --feature_module=residual_${head_num1}
# done &

# for head_num2 in "${head_num_layers2[@]}";do
# CUDA_VISIBLE_DEVICES=1 python3 QCS_hcm.py --learning_rate=1e-5 --batch_size=128 --n_epochs=200 --world_size=1 --weight_decay=1e-4 \
#             --guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e --dataset_path=../data/RAF-DB_balanced --dataset_name=RAF-DB \
#             --use_sampler=True --model_type=ir50 --num_workers=32 --use_tf=True --loss=CE --pin_memory=True --learnable_input_dist=True --freeze_backbone=True \
#             --feature_module=residual_${head_num2}
# done &

# wait 


for model_type in ir50_1 ir50_2; do
CUDA_VISIBLE_DEVICES=0 python3 QCS_hcm.py --learning_rate=1e-5 --batch_size=128 --n_epochs=200 --world_size=1 --weight_decay=1e-4 \
--guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e --dataset_path=../data/RAF-DB_balanced --dataset_name=RAF-DB \
--use_sampler=True --model_type=$model_type --num_workers=32 --use_tf=True --loss=CE --pin_memory=True --learnable_input_dist=True  --remain_backbone=True
done &



for model_type in ir50_1 ir50_2; do
CUDA_VISIBLE_DEVICES=1 python3 QCS_hcm.py --learning_rate=1e-5 --batch_size=128 --n_epochs=200 --world_size=1 --weight_decay=1e-4 \
--guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e --dataset_path=../data/RAF-DB_balanced --dataset_name=RAF-DB \
--use_sampler=True --model_type=$model_type --num_workers=32 --use_tf=True --loss=CE --pin_memory=True --learnable_input_dist=True 
done &

wait 