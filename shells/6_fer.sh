
torchrun --nproc_per_node=3 adv_training.py --world_size=3 --use_tf=True \
--model_type=fmae_small \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced \
--id_strategy=masking --n_blocks=100 --detach_lowlevel=True --beta=0 --partial_update=True \
--learning_rate=1e-5 --batch_size=192 --n_epochs=200 --use_sampler=True \
