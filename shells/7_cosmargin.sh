CUDA_VISIBLE_DEVICES=0 python3 pretrain.py --world_size=1 \
--learning_rate=0.000003 --batch_size=128 \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--n_epochs=20  \
--name cos_margin_3e-6 --quality_model_path=checkpoint/quality \
--instance_adaloss_ckpt=checkpoint/vanilla12m \
--proportion_alpha=0 \
--cos_margin_loss=True &

CUDA_VISIBLE_DEVICES=1 python3 pretrain.py --world_size=1 \
--learning_rate=0.000001 --batch_size=128 \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--n_epochs=20  \
--name cos_margin_1e-6 --quality_model_path=checkpoint/quality \
--instance_adaloss_ckpt=checkpoint/vanilla12m \
--proportion_alpha=0 \
--cos_margin_loss=True

