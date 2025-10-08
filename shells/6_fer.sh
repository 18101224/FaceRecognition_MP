torchrun --nproc_per_node=3 FER_CL.py --world_size=3 --num_workers=32 --use_tf=True \
--learning_rate=1.5e-6 --batch_size=128 --n_epochs=30 --weight_decay=5e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet8 --num_classes=8 --use_sampler=True --img_size=112 \
--model_type=ir50 --feature_branch=True --use_bn=True \
--loss=EKCL --k_meeting_dist=0.1 --kcl_k=5 --beta=0.3  --k_meeting=2_3 --temperature=0.1 --utilize_target_centers=True --balanced_cl=True --k_grad=True 