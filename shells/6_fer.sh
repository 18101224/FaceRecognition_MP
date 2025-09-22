

CUDA_VISIBLE_DEVICES=0 python3 MoCo.py --learning_rate=1e-5 --batch_size=64 --n_epochs=200 --world_size=1 --num_workers=32 --weight_decay=5e-4 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --img_size=112 \
--model_type=ir50 \
--loss=CE &

CUDA_VISIBLE_DEVICES=2 python3 MoCo.py --learning_rate=1e-5 --batch_size=64 --n_epochs=200 --world_size=1 --num_workers=32 --weight_decay=5e-4 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --img_size=112 \
--model_type=ir50 \
--mean_weight=checkpoint/ir50_72k_wosampler \
--loss=KBCL --kcl_k=5 --beta=0.3 --temperature=0.1 --utilze_class_centers=True --moco_k=72 &

wait 




