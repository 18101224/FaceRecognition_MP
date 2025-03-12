#python3 finetune.py --world_size=1 \
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 finetune.py --world_size=2 \
--data_path=../data/AffectNet7 --learning_rate=0.0001 \
--ckpt_path=checkpoint/RL0 \
--idx=1 --log_file=log.xlsx --batch_size=1024 --n_epochs=200 \
--bs_list 1024 --lr_list 0.0001 \
--name=RL0 \
--mode=r
