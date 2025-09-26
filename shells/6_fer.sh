torchrun --nproc_per_node=2 MoCo.py --world_size=2 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=32 --n_epochs=200 --weight_decay=5e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 --use_sampler=True --img_size=112 --use_view=True \
--model_type=kprpe12m \
--loss=BCL --utilze_target_centers=True