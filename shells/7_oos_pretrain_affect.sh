CUDA_VISIBLE_DEVICES=0 python3 OOS_pretraining.py --world_size=1 \
    --dataset_path=../data/clothing1m --dataset_name=clothing1m --n_folds=4 --random_seed=566 --weight_decay=5e-4 \
    --learning_rate=1e-4 --batch_size=256 --n_epochs=200 --architecture=resnet50 --num_classes=14 --use_tf=True --num_workers=32 