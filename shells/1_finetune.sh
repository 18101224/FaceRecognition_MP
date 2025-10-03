python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=64 --n_epochs=50 --weight_decay=5e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet8 --num_classes=8 --use_sampler=True --img_size=112 \
--model_type=ir50 \
--resume_path=checkpoint/149691721-60d7-4bce-8e2f-02edd7f06da8 \
--loss=CE_ETF --etf_statistics=True --etf_std=0.3 --etf_weight=2