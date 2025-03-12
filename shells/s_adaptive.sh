CUDA_VISIBLE_DEVICES=0,1 /pscratch/sd/s/sgkim/hcir/condaenvs/cv2/bin/torchrun --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 --nnodes=1 pretrain.py --world_size=2 \
--learning_rate=0.0003 --batch_size=128 \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--n_epochs=20  \
--LDAM=True --ldam_weight=0.1 \
--cos_margin_loss=True --margin=0.3 \
--name only_margin_high --quality_model_path=checkpoint/quality \
--instance_adaloss_ckpt=checkpoint/vanilla12m \
--proportion_alpha=0 > log1.txt 2>&1 &
