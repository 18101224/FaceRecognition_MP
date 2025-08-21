  
  
  

CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:2950 train_imbalanced.py \
     --learning_rate=0.1 --batch_size=256 --n_epochs=90 --weight_decay=5e-4 --world_size=3 \
     --wandb_token=../wandb.txt --server=pm --dataset_name=imagenet_lt --dataset_path=/data1/imgnet \
     --cosine_scaling=32 --cosine_constant_margin=0.5 --model_type=resnet50 --loss=CE_BCL --aug=True \
     --randaug_n=2 --randaug_m=10 --use_warmup=True --ce_weight=2 --cl_weight=0.6 --ece_weight=1 --cutout=True 