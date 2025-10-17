CUDA_VISIBLE_DEVICES=1 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=64 --n_epochs=30 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=ir50 --feature_branch=True --use_bn=True \
--loss=KBCL --kcl_k=10 --temperature=0.1 --utilize_target_centers=True --moco_k=96 --balanced_cl=True &

CUDA_VISIBLE_DEVICES=2 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=64 --n_epochs=30 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=ir50 --feature_branch=True --use_bn=True \
--loss=KBCL --kcl_k=10 --temperature=0.1 --utilize_target_centers=True --moco_k=96 --balanced_cl=True &

wait 

CUDA_VISIBLE_DEVICES=1 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=64 --n_epochs=30 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=ir50 --feature_branch=True --use_bn=True \
--loss=KBCL_ETF --kcl_k=10 --temperature=0.1 --utilize_target_centers=True --moco_k=96 --balanced_cl=True --etf_weight=1 &

CUDA_VISIBLE_DEVICES=2 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=64 --n_epochs=30 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=ir50 --feature_branch=True --use_bn=True \
--loss=KBCL_ETF --kcl_k=10 --temperature=0.1 --utilize_target_centers=True --moco_k=96 --balanced_cl=True --etf_weight=1 &