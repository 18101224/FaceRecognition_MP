python3 FER_CL.py --world_size=1 --num_workers=32 \
--learning_rate=1e-5 --batch_size=16 --n_epochs=200 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=kprpe12m \
--loss=BEAC --beta=2 --utilize_target_centers=True