CUDA_VISIBLE_DEVICES=0 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=256 --n_epochs=30 --weight_decay=5e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet8 --num_classes=8 --use_sampler=True --img_size=112 \
--model_type=ir50 \
--loss=CE_ETF --etf_weight=1 &

CUDA_VISIBLE_DEVICES=1 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=256 --n_epochs=30 --weight_decay=5e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet8 --num_classes=8 --use_sampler=True --img_size=112 \
--model_type=ir50 \
--loss=CE_ETF --etf_weight=0.5 & 

CUDA_VISIBLE_DEVICES=2 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=256 --n_epochs=30 --weight_decay=5e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet8 --num_classes=8 --use_sampler=True --img_size=112 \
--model_type=ir50 \
--loss=CE_ETF --etf_weight=3 &