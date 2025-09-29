CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 FER_CL.py --world_size=2 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=256 --n_epochs=30 --weight_decay=5e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 --img_size=112 --use_sampler=True \
--model_type=kprpe12m \
--loss=CE_ETF --etf_weight=2 