CUDA_VISIBLE_DEVICES=0 python3 MoCo.py --world_size=1 --use_tf=True --num_workers=32 \
--learning_rate=1e-5 --batch_size=64 --n_epochs=200 --weight_decay=5e-4 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --use_sampler=True \
--model_type=kp_rpe \
--loss=CE