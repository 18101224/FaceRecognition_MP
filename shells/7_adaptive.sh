LEARNING_RATES=(0.0001 0.00001 0.00003)

for LR in "${LEARNING_RATES[@]}"; do
CUDA_VISIBLE_DEVICES=0 python3 train_imbalanced.py --world_size=1 --learning_rate=$LR \
                        --batch_size=128 --n_epochs=200 --weight_decay=5e-4 --scheduler=cosine --cos=True --num_workers=4 \
                        --wandb_token=../wandb.txt --cosine_scaling=16 --loss=BCL --model_type=ir50 --dataset_name=RAF-DB \
                        --ce_weight=1 --cl_weight=0.35 --feature_branch=True --dataset_path=../data/RAF-DB --aug=True --cutout=True \
                        --use_wandb=True
wait
done