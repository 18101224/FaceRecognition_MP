CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 FER_CL.py --world_size=2 --num_workers=32 --use_tf=True \
--learning_rate=1e-5 --batch_size=256 --n_epochs=200 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=kprpe12m --feature_branch=True --use_bn=True \
--loss=KBCL_ETF --kcl_k=32 --beta=0.3 --temperature=0.1 --moco_k=288 --balanced_cl=True --etf_weight=1  --utilize_target_centers=True