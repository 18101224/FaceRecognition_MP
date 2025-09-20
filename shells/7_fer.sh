CUDA_VISIBLE_DEVICES=2 python3 MoCo.py --learning_rate=1e-5 --batch_size=64 --n_epochs=200 --world_size=1 --num_workers=32 --weight_decay=5e-4 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --use_sampler=True \
--model_type=Pyramid_ir50 \
--loss=CE  --img_size=224