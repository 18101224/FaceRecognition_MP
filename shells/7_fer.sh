

CUDA_VISIBLE_DEVICES=0 python3 QCS_hcm.py \
  --dataset_path=../data/RAF-DB_balanced \
  --dataset_name=RAF-DB \
  --guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e \
  --num_workers=32 \
  --world_size=1 \
  --batch_size=4 \
  --n_epochs=200 \
  --learning_rate=1e-4 \
  --use_sampler=True \
  --model_type=ir50 --use_hcm=True 
