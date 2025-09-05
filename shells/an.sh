python3 analyze_classifier.py --dataset_name=RAF-DB \
 --model_paths checkpoint/07db3fe34-a755-44e1-9d37-cffc175f2d71 --save_path="results/RAF-DB_CE_IR50_WO_sampler" \
 --dataset_path=../data/RAF-DB_balanced --mode=analsys --ckpt_type=best_acc --model_type=ir50