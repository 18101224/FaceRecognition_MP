for lr in 1.5e-6 2e-6 3e-6 8e-7; do
CUDA_VISIBLE_DEVICES=0 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=$lr --batch_size=64 --n_epochs=30 --weight_decay=5e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet8 --num_classes=8 --use_sampler=True --img_size=112 \
--mean_weight=checkpoint/af8_ir50_fb \
--model_type=ir50 --feature_branch=True --use_bn=True \
--loss=KBCL --kcl_k=5 --beta=0.3 --temperature=0.1 --utilize_target_centers=True --moco_k=72 --balanced_cl=True &

CUDA_VISIBLE_DEVICES=1 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=$lr --batch_size=64 --n_epochs=30 --weight_decay=5e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet8 --num_classes=8 --use_sampler=True --img_size=112 \
--mean_weight=checkpoint/af8_ir50_fb \
--model_type=ir50 --feature_branch=True  \
--loss=KBCL --kcl_k=5 --beta=0.3 --temperature=0.1 --utilize_target_centers=True --moco_k=72 --balanced_cl=True &

CUDA_VISIBLE_DEVICES=2 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=$lr --batch_size=64 --n_epochs=30 --weight_decay=5e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet8 --num_classes=8 --img_size=112 \
--mean_weight=checkpoint/af8_ir50_fb \
--model_type=ir50 --feature_branch=True \
--loss=KBCL --kcl_k=5 --beta=0.3 --temperature=0.1 --utilize_target_centers=True --moco_k=72 --balanced_cl=True &
wait 
done