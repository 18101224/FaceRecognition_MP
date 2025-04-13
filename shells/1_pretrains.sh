python3 pretrain.py --world_size=1 \
--learning_rate=0.0001 --batch_size=256 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--n_epochs=50  \
--name cos_vanilla_raf1e-5 --quality_model_path=checkpoint/quality \
--proportion_alpha=0 \
--instance_adaloss_ckpt=checkpoint/vanilla12m \
--server=1

