CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:29000 --nnodes=1 clothing_train.py --world_size=2 \
--learning_rate=0.00001 --batch_size=1024 --n_epochs=50 \
--dataset_path=../data/clothing1m --dataset_name=clothing \
--wandb_token=../wandb.txt --server=5 --name=label_loss1e-5 \
--lq_ckpt=checkpoint/vanilla5e-6 --use_dropout=True --adaptive_margin=0.3 \
--constant_margin=0.4 &


CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --rdzv-backend=c10d --rdzv-endpoint=localhost:29001 --nnodes=1 clothing_train.py --world_size=2 \
--learning_rate=0.000005 --batch_size=1024 --n_epochs=50 \
--dataset_path=../data/clothing1m --dataset_name=clothing \
--wandb_token=../wandb.txt --server=5 --name=label_loss_5e-6 \
--lq_ckpt=checkpoint/vanilla5e-6 --use_dropout=True --adaptive_margin=0.3 \
--constant_margin=0.4




