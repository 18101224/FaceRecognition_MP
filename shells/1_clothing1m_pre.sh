#LEARNING_RATES=(0.01 0.05 )
LEARNING_RATES=(0.001 0.005)
for LR in "${LEARNING_RATES[@]}"; do
CUDA_VISIBLE_DEVICES=0 python3 OOS_pretraining.py --world_size=1 \
                            --dataset_path=../data/clothing1m --dataset_name=clothing1m --random_seed=566 \
                            --learning_rate=$LR --batch_size=128 --n_epochs=80 --wandb_token=../wandb.txt --server=1 \
                            --architecture=resnext50 --num_classes=14 --cos_constant_margin=0.5 --n_folds=5 --weight_decay=5e-2 --pretrained=True --cos_scaling=16
done

python3 -c "from utils import send_message; send_message('Clothing1M Pretraining finished')"