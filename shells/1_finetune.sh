for k in 72 256 1024 8192 ; do 
python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-5 --batch_size=64 --n_epochs=200 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=CAER --dataset_path=../data/CAER-S --num_classes=7 --img_size=112 --use_sampler=True --imb_factor=0.1 \
--model_type=ir50 --feature_branch=True --use_bn=True \
--loss=KBCL --kcl_k=10 --beta=0.3 --temperature=0.1 --utilize_target_centers=True --moco_k=$k --balanced_cl=True
done

