CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 FER_CL.py --world_size=3 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=32 --n_epochs=30 --weight_decay=5e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=kprpe12m \
--loss=EKCL_ETF --etf_weight=2 --kcl_k=5 --beta=0.3 --temperature=0.1 --balanced_cl=True  --utilize_target_centers=True