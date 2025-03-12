python3 pretrain.py --world_size=1 \
--learning_rate=0.00001 --batch_size=96   \
--dataset_name=RAF --dataset_path=../data/RAF-DB \
--use_hf=True --token_path=HF.txt \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface4m \
--n_epochs=200 \
--name=kprpeDropout12m \

#--force_download=True \