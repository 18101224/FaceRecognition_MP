CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 OOS_pretraining.py --world_size=2 --batch_size=256 --n_epochs=30 --learning_rate=5e-06 --weight_decay=5e-4 \
    --dataset_path=../data/AffectNet7 --dataset_name=AffectNet --random_seed=566 --wandb_token=../wandb.txt --kprpe_ckpt_path=checkpoint/adaface_vit_base_kprpe_webface12m \
    --num_classes=7 --cos_constant_margin=0.5 --n_folds=3 --server=7 
