


ks1=(16 32 64 128 256 512)
moco_ks=(256 512 768 1024 2048)

for k in "${ks1[@]}"; do
  for moco_k in "${moco_ks[@]}"; do
    echo "=== Running with kcl_k=$k, moco_k=$moco_k ==="
    python3 FER_CL.py --world_size=1 --num_workers=32 \
      --learning_rate=1e-5 --batch_size=256 --n_epochs=30 --weight_decay=1e-4 --optimizer=AdamW --scheduler=cosine \
      --dataset_name=CINIC10 --dataset_path=../data/cinic10 --num_classes=10 --img_size=224 --imb_factor=0.01 --use_sampler=True \
      --model_type=Dinov2 --feature_branch=True --use_bn=True \
      --loss=KBCL --beta=1 --temperature=0.1 --moco_k="$moco_k" --kcl_k="$k" --utilize_target_centers=True
  done
done

python3 -c"from utils.pushover import send_message; send_message('Cinic_CE finished')"