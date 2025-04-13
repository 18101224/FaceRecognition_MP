
#CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:29500 --nnodes=1 pretrain.py --world_size=2 \
#--learning_rate=0.0003 --batch_size=128 \
#--dataset_name=AffectNet --dataset_path=../data/AffectNet7 \
#--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
#--n_epochs=20  \
#--LDAM=True --ldam_weight=0.1 \
#--cos_margin_loss=True --margin=0.3 \
#--name only_margin_high --quality_model_path=checkpoint/quality \
#--instance_adaloss_ckpt=checkpoint/vanilla12m \
#--proportion_alpha=0 > log1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:29501 --nnodes=1 pretrain.py --world_size=2 \
--learning_rate=0.00003 --batch_size=128 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--n_epochs=200  \
--name cos_adaptive_margin3e-5_m4 --quality_model_path=checkpoint/quality \
--instance_adaloss_ckpt=checkpoint/cos_constant_margin_raf1e-5 \
--instance_ada_loss=True \
--server=5 &

CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:29502 --nnodes=1 pretrain.py --world_size=2 \
--learning_rate=0.000007 --batch_size=128 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--n_epochs=200  \
--name cos_adaptive_margin1e-7_m4 --quality_model_path=checkpoint/quality \
--instance_adaloss_ckpt=checkpoint/cos_constant_margin_raf1e-5 \
--instance_ada_loss=True \
--server=5

