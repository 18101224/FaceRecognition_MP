CUDA_VISIBLE_DEVICES=0,1,2 \
torchrun --nproc_per_node=3 RL.py --world_size=3 \
--learning_rate=0.00001 \
--name=rl_training_with_ema_init \
--wandb_token=../wandb.txt \
--n_epochs=200 \
--batch_size=1536 \
--dataset=../data/AffectNet7 \
--env_init_path=checkpoint/raf_classifier.pth \
--gamma=0 \
--k=1 \
--save_path=5 \
--ada_loss=True \
--init_epoch=5 \
#--gp=0.05 \

