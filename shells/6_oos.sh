lrs=(5e-6 7e-6)

for lr in "${lrs[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 OOS_target_train.py --world_size=4 \
        --n_folds=3 --g_net_ckpt=checkpoints/6062011123857 --random_seed=566 --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
        --cos_constant_margin=0.5 --instance_ada_loss=True \
        --learning_rate=$lr --batch_size=256 --n_epochs=30 \
        --dataset_path=../data/AffectNet7 --dataset_name=AffectNet7 \
        --wandb_token=../wandb.txt --server=5 \
        --confidence_constant=1 --cos_scaling=16 \
        --oos_tensor=checkpoint/conf_db.pt
done

python3 -c "from utils import send_message; send_message('server5 all experiment done')"