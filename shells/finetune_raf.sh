#torchrun --nproc_per_node=3 finetune.py --world_size=3 \
python3 finetune.py --world_size=1 \
--data_path=../data/RAF-DB --learning_rate=0.00001 \
--ckpt_path=checkpoint/RL0 \
--idx=0 --log_file=log.xlsx --batch_size=256 --n_epochs=200 \
--bs_list 256 --lr_list 0.00001 0.00005 0.0001 0.0005 \
--name=RL0 \
--mode=r
