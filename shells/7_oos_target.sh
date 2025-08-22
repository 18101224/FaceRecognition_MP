

for LR in 0.000003 0.000006; do
CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:23501 OOS_target_train.py --world_size=2 \
        --n_folds=3 --g_net_ckpt=checkpoints/7080801341664 --random_seed=566 --kprpe_ckpt_path=checkpoint/adaface_vit_base_kprpe_webface12m \
        --cos_constant_margin=0.5 --instance_ada_loss=True \
        --learning_rate=$LR --batch_size=256 --n_epochs=30 \
        --dataset_path=../data/AffectNet7 --dataset_name=AffectNet \
        --wandb_token=../wandb.txt --server=pm \
        --confidence_constant=1 --use_tf=True --cos_scaling=16  \
        --oos_tensor=checkpoint/Affect.pt 
done
wait
