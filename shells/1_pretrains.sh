
python3 pretrain.py --world_size=1 \
      --learning_rate=0.000007 --batch_size=84   \
      --dataset_name=AffectNet --dataset_path=../data/AffectNet7 \
      --wandb_token=../wandb.txt \
      --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
      --n_epochs=20 \
      --name=vanilla12m \


python3 pretrain.py --world_size=1 \
      --learning_rate=0.000007 --batch_size=84   \
      --dataset_name=AffectNet --dataset_path=../data/AffectNet7 \
      --wandb_token=../wandb.txt \
      --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface4m \
      --n_epochs=20 \
      --name=vanilla4m \
