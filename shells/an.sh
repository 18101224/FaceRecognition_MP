python3 analyze_classifier.py --dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--save_path=results/KBCL_ETF_kprpe --model_paths checkpoint/kprpe_KBCL_ETF checkpoint/kprpe_CE --model_names "KBCL_ETF" "CE" --mode=compare --ckpt_type=best


