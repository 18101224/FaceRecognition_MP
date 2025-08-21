lrs=(1e-3 1e-4 5e-3)

for lr in "${lrs[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python3 OOS_target_train.py --world_size=1 \
        --n_folds=5 --g_net_ckpt=checkpoints/clothing1m_pretrained --random_seed=566 --cos_constant_margin=0.5 --instance_ada_loss=True \
        --learning_rate=$lr --batch_size=128 --n_epochs=80 \
        --dataset_path=../data/clothing1m --dataset_name=clothing1m \
        --wandb_token=../wandb.txt --server=7 \
        --confidence_constant=1  \
        --oos_tensor=checkpoint/clothing1m_conf_db.pt --architecture=resnext50 --weight_decay=5e-2 --num_classes=14 --cos_scaling=16 --pretrained=True
done

python3 -c 'from utils import send_message; send_message("oos target clothing1m done")'