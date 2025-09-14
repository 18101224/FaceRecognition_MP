python3 adv_training.py --world_size=1 \
    --dataset_name='RAF-DB' --dataset_path='../data/RAF-DB_balanced' \
    --id_strategy=masking --n_blocks=96 --detach_lowlevel=True --beta=0.3 --partial_update=True \
    --learning_rate=1e-5 --batch_size=36 --n_epochs=200 --use_tf=True --use_sampler=True \
    --model_type=ir50 --debug=True