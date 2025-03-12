python3 pretrain.py --world_size=1 \
--learning_rate=0.000007 --batch_size=84   \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 \
--wandb_token=../wandb.txt --kp_rpe_cfg_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--n_epochs=10 \
--ckpt=checkpoint/kprpeDropout20_webface12mFlipFalseLearnlow/ \
--name=kprpeDropout20_webface12mAffectNetClassAdaDropout \
--class_ada_loss=True \
--class_alpha=0.3 \
--instance_ada_dropout=True \
--quality_model_path=checkpoint/quality


