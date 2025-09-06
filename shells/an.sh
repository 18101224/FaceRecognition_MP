prefix="checkpoint/"
python3 analyze_classifier.py --dataset_name=cifar100 \
 --dataset_path=../data --save_path=results/big_table_cifar10 --model_paths "$prefix"pm9f741a0d-a8ad-42bb-ae11-29f341d0d711 "$prefix"pmd7e56df0-2933-4e95-9d0e-c645aaa6fb90 \
 --model_names 'mean+std loss' 'std only loss' --imb_factor=0.01 --mode=compare --ckpt_type best_acc