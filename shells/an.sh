prefix="checkpoint/"
python3 analyze_classifier.py --dataset_name=cifar100 \
 --dataset_path=../data --save_path=results/big_table_cifar100 --model_paths "$prefix"pm90b3bfb1-fceb-42aa-86f2-0d87faa6c1f9 "$prefix"pmc5a7410c-a568-4d0f-aec5-ae459e62b21b \
 --model_names 'quadratic loss' 'std only loss' --imb_factor=0.01 --mode=compare --ckpt_type best_acc