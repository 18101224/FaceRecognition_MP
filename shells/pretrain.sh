CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 pretrain.py --world_size=4 \
--learning_rate=0.0001 --batch_size=256   \
--dataset_name=RAF --dataset_path=../data/RAF-DB \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface4m \
--n_epochs=200 \
--name=kprpeDropout
