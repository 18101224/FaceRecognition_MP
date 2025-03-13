#CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 --nnodes=1 pretrain.py --world_size=2 \
#--learning_rate=0.000001 --batch_size=512 \
#--dataset_name=AffectNet --dataset_path=../data/AffectNet7 \
#--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
#--n_epochs=200  \
#--name only_margin_high5e4 --quality_model_path=checkpoint/quality \
#--instance_adaloss_ckpt=checkpoint/vanilla12m \
#--proportion_alpha=0

CUDA_VISIBLE_DEVICES=2 python3 pretrain.py --world_size=1 \
--learning_rate=0.000001 --batch_size=128 \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--n_epochs=200  \
--name only_margin_low1e-6 --quality_model_path=checkpoint/quality \
--instance_adaloss_ckpt=checkpoint/vanilla12m \
--proportion_alpha=0 &

CUDA_VISIBLE_DEVICES=3 python3 pretrain.py --world_size=1 \
--learning_rate=0.000005 --batch_size=128 \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--n_epochs=200  \
--name only_margin_low5e-6 --quality_model_path=checkpoint/quality \
--instance_adaloss_ckpt=checkpoint/vanilla12m \
--proportion_alpha=0
