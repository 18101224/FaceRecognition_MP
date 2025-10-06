
CUDA_VISIBLE_DEVICES=0 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-5 --batch_size=64 --n_epochs=200 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=CAER --dataset_path=../data/CAER-S --num_classes=7 --use_sampler=True --img_size=112 --imb_factor=0.1 --use_sampler=True \
--model_type=ir50 \
--loss=CE_ETF --etf_weight=1

CUDA_VISIBLE_DEVICES=0 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-5 --batch_size=64 --n_epochs=200 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=CAER --dataset_path=../data/CAER-S --num_classes=7 --use_sampler=True --img_size=112 --imb_factor=0.1 --use_sampler=True \
--model_type=ir50  \
--mean_weight=checkpoint/caer_ir50_ce_etf \
--loss=KBCL --kcl_k=32 --beta=0.3 --temperature=0.1 --utilize_target_centers=True --moco_k=72 --balanced_cl=True 


CUDA_VISIBLE_DEVICES=0 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-5 --batch_size=64 --n_epochs=200 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=CAER --dataset_path=../data/CAER-S --num_classes=7 --use_sampler=True --img_size=112 --imb_factor=0.1 --use_sampler=True \
--model_type=ir50 --feature_branch=True \
--mean_weight=checkpoint/caer_ir50_kbcl_fb \
--loss=KBCL --kcl_k=32 --beta=0.3 --temperature=0.1 --utilize_target_centers=True --moco_k=72 --balanced_cl=True 

CUDA_VISIBLE_DEVICES=0 python3 FER_CL.py --world_size=1 --num_workers=32 --use_tf=True \
--learning_rate=1e-5 --batch_size=64 --n_epochs=200 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=CAER --dataset_path=../data/CAER-S --num_classes=7 --use_sampler=True --img_size=112 --imb_factor=0.1 --use_sampler=True \
--model_type=ir50 --feature_branch=True \
--mean_weight=checkpoint/caer_ir50_kbcl_etf \
--loss=KBCL_ETF --kcl_k=32 --beta=0.3 --temperature=0.1 --utilize_target_centers=True --moco_k=72 --balanced_cl=True --etf_weight=1
