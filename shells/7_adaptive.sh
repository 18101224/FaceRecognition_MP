CUDA_VISIBLE_DEVICES=1 python3  pretrain.py --world_size=1 \
--learning_rate=0.000001 --batch_size=128 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--n_epochs=300  \
--name constant_margin+weight_loss1e-6 --quality_model_path=checkpoint/quality \
--instance_adaloss_ckpt=checkpoint/vanilla12m \
--cos_constant_margin=0.4 \
--angle_loss=0.5 \
--server=7  &

CUDA_VISIBLE_DEVICES=2 python3  pretrain.py --world_size=1 \
--learning_rate=0.00001 --batch_size=128 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--n_epochs=300  \
--name constant_margin+weight_loss1e-5 --quality_model_path=checkpoint/quality \
--instance_adaloss_ckpt=checkpoint/vanilla12m \
--cos_constant_margin=0.4 \
--angle_loss=0.5 \
--server=7