python3 analyze_classifier.py --dataset_name=AffectNet --dataset_path=../data/AffectNet8 --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--save_path=results/AffectNet8 --model_paths checkpoint/af8_kprpe_ce checkpoint/af8_ce_etf --model_names "CE" "CE_ETF" --mode=compare --ckpt_type=best --num_classes=8


