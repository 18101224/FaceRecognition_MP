




torchrun --nproc_per_node=3 --rdzv_backend=c10d --rdzv_endpoint=localhost:23501 OOS_target_train.py --world_size=3 \
  --learning_rate=0.0001 --batch_size=128 --pretrained=True --n_epochs=80 --architecture=resnext50 \
  --dataset_path=../data/clothing1m --dataset_name=clothing1m --num_classes=14 \
  --cos_constant_margin=0.5 --num_workers=32 \
  --n_folds=5 --g_net_ckpt=checkpoints/clothing_best --random_seed=566 --oos_tensor=checkpoint/clothing1m_conf_db.pt