
for beta in 0.1 0.3 1 2; do
python3 FER_CL.py --world_size=1 --num_workers=32 \
--learning_rate=1e-5 --batch_size=64 --n_epochs=100 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=CAER --dataset_path=../data/CAER-S --num_classes=7 --use_sampler=True --img_size=112 --imb_factor=0.1 \
--model_type=ir50 \
--loss=KBCL_ETF --temperature=0.1 --utilize_target_centers=True --beta=$beta --moco_k=256 --kcl_k=32 --etf_weight=1
done 