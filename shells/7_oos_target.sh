lrs=(5e-6)

for lr in "${lrs[@]}"; do
    CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 --master_port=29503 OOS_target_train.py --world_size=2 \
        --n_folds=3  --g_net_ckpt=checkpoints/7080801341664 --random_seed=566 --kprpe_ckpt_path=checkpoint/adaface_vit_base_kprpe_webface12m \
        --cos_constant_margin=0.5 --instance_ada_loss=True \
        --learning_rate=$lr --batch_size=256 --n_epochs=30 \
        --dataset_path=../data/AffectNet7 --dataset_name=AffectNet \
        --wandb_token=../wandb.txt --server=7 \
        --confidence_constant=1 --cos_scaling=16  \
        --oos_tensor=checkpoint/conf_db_AffectNet.pt --as_bias=True --num_classes=7 --weight_decay=5e-2
done

python3 -c 'from utils import send_message; send_message("oos target w.o. dropout done")'
