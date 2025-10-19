CUDA_VISIBLE_DEVICES=0 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=144 --n_epochs=30 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=ir50 --feature_branch=True --use_bn=True \
--loss=EKCL --num_clusters 80 200 --sizes_clusters 3 5 --balanced_cl=True --batch_pairs_only=True --utilize_target_centers=True & 

CUDA_VISIBLE_DEVICES=1 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=144 --n_epochs=30 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=ir50 --feature_branch=True --use_bn=True \
--loss=EKCL --num_clusters 80 200 --sizes_clusters 3 5 --balanced_cl=True --batch_pairs_only=True --utilize_target_centers=True & 

CUDA_VISIBLE_DEVICES=2 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-6 --batch_size=144 --n_epochs=30 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=ir50 --feature_branch=True --use_bn=True \
--loss=EKCL --num_clusters 80 200 --sizes_clusters 3 5 --balanced_cl=True --batch_pairs_only=True --utilize_target_centers=True & 

wait 

pythn3 -c "from utils.pushover import send_message; send_message(' s7 finished')"