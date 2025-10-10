python3 analyze_classifier.py --dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 \
 --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--save_path=results/raf_compare --mode=compare --model_type=kprpe12m --model_paths checkpoint/raf-kprpe_ce checkpoint/raf-kprpe-kbcl  \
--model_names CE KBCL_ETF --ckpt_type=best