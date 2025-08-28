

for lr in 0.15 0.1 ; do
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train_imbalanced.py  --world_size=2 \
        --batch_size=256 --n_epochs=200 --weight_decay=5e-4 \
          --cos=True --momentum=0.9 --world_size=1 \
        --model_type=e2_resnet32  --imb_type=exp --imb_factor=0.01 \
        --dataset_path=../data --aug=True --cutout=True --use_wandb=True  --feature_branch=True --use_tf=True --num_workers=16\
         --cosine_scaling=32    --temperature=0.1 --scheduler=warmup --learning_rate=$lr --dataset_name=cifar100 --use_mean=True --loss=BCL --ce_weight=1 --cl_weight=1
wait 
done 
wait 