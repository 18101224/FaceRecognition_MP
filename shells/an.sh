python3 analyze_classifier.py --dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 \
 --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--save_path=results/raf_kbcl_macro --mode=analysis --model_type=kprpe12m --model_paths checkpoint/raf_kprpe_kbcl_best_macro \
--model_names KBCL_Macro --ckpt_type=best