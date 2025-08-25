make_cmd () {

  local EXTRA=$1        # loss·weight·스케줄 인자 묶음
   $EXTRA
}


for weight in 0.3 0.5 1; do

  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3  train_imbalanced.py --world_size=3 \
        --batch_size=256 --n_epochs=200 --weight_decay=5e-4 \
        --cos=True --momentum=0.9 --world_size=1 --wandb_token=../wandb.txt \
        --model_type=e2_resnet32  --imb_type=exp --imb_factor=0.01 \
        --dataset_path=../data --aug=True --cutout=True --use_wandb=True  --feature_branch=True  --regular_simplex=True --feature_module=deepcomplex_3  \
         --cosine_scaling=32   --temperature=0.1 --scheduler=warmup  --learning_rate=0.15 --dataset_name=cifar10  --loss=BCL_ECE --ece_weight=$weight 

  wait
done
