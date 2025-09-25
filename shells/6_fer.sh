CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 MoCo.py --world_size=3 --num_workers=32 --use_tf=True \
--learning_rate=1e-5 --batch_size=64 --n_epochs=200 --weight_decay=1e-5 \
--dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 --img_size=112 --use_sampler=True \
--model_type=ir50 \
--loss=EKCL_ETF --kcl_k=5 --beta=0.3 --temperature=0.1 --utilze_class_centers=True --k_grad=True --etf_weight=2 --balanced_cl=True