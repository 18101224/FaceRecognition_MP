
CUDA_VISIBLE_DEVICES=0,1 /home/gpuadmin/anaconda3/envs/main/bin/torchrun --nproc_per_node=2 QCS_hcm.py --world_size=2  \
  --dataset_path=../data/RAF-DB_balanced \
  --dataset_name=RAF-DB \
  --guide_path=checkpoint/pme3c5fe43-a9ec-4a37-8ea4-f11268a0ec6e \
  --num_workers=32 \
  --batch_size=128 \
  --n_epochs=200 \
  --learning_rate=1e-5 \
  --use_sampler=True \
  --model_type=kp_rpe --loss=CE --cl_weight=0.3 

