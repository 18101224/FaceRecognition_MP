
make_cmd () {

  local EXTRA=$1        # loss·weight·스케줄 인자 묶음
  CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 train_imbalanced.py \
        --batch_size=64 --n_epochs=200 --weight_decay=5e-4 \
          --cos=True --momentum=0.9 --world_size=2 \
        --model_type=ir50  --imb_type=exp --imb_factor=0.01 \
        --dataset_path=../data/RAF-DB_aligned --dataset_name=RAF-DB --use_sampler=True --aug=True --cutout=True --use_wandb=True  --feature_branch=True --use_tf=True --num_workers=32 \
         --cosine_scaling=32  --temperature=0.1 --scheduler=warmup  $EXTRA
}

make_cmd "--learning_rate=${3e-5} --loss=BCL --cl_weight=1 --ce_weight=1" 
