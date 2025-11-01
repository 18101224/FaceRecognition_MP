

# python3 analyze_classifier.py --dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 \
#  --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/raf-kbcl-9250 --mode=analysis --model_type=kprpe12m --model_paths checkpoint/raf-kbcl-9250 \
# --model_names KBCL_Macro --ckpt_type=best



# python3 analyze_classifier.py --dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 \
#  --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/raf-ce-w_sampler-9243 --mode=analysis --model_type=kprpe12m --model_paths checkpoint/raf-ce-w_sampler-9243 \
# --model_names CE --ckpt_type=best

raf_kbcl_etf="raf-kbcl_etf_9224"
raf_kbcl="raf-kbcl-9250"
raf_ce="raf-ce-w_sampler-9243"
af7_kbcl_etf="af7-kbcl_etf_6854"
af7_ce="af7-ce_6800"
af7_kbcl="af7-kbcl_6811"
caer_kbcl="caer-kbcl"
caer_kbcl_etf="caer-kbcl_etf"
caer_ce="caer-ce"


# python3 analyze_classifier.py --dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 \
# --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/af7 --mode=dataset --model_type=ir50 --imb_factor=0.1 --model_paths checkpoint/$caer_kbcl_etf \
# --model_names KBCL_ETF --ckpt_type=best

# python3 analyze_classifier.py --dataset_name=CAER --dataset_path=../data/CAER-S --num_classes=7 \
# --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/caer --mode=dataset --model_type=ir50 --imb_factor=0.1 --model_paths checkpoint/$caer_kbcl \
# --model_names KBCL --ckpt_type=best

# python3 analyze_classifier.py --dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 \
# --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/raf-db --mode=dataset --model_type=ir50 --imb_factor=0.1 --model_paths checkpoint/$caer_ce \
# --model_names CE --ckpt_type=best

# python3 analyze_classifier.py --dataset_name=CAER --dataset_path=../data/CAER-S --num_classes=7 \
# --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/caer_compare_kbcl_etf_vs_kbcl_vs_ce --mode=compare --model_type=ir50 --model_paths checkpoint/$caer_ce checkpoint/$caer_kbcl checkpoint/$caer_kbcl_etf \
# --model_names CE KBCL KBCL_ETF --ckpt_type=best

# python3 analyze_classifier.py --dataset_name=CAER --dataset_path=../data/CAER-S --num_classes=7 \
# --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/caer_kbcl_etf_an --mode=analysis --model_type=ir50 --imb_factor=0.1 --model_paths checkpoint/$caer_kbcl_etf \
# --model_names KBCL_ETF --ckpt_type=best

# python3 analyze_classifier.py --dataset_name=CAER --dataset_path=../data/CAER-S --num_classes=7 \
# --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/caer_kbcl_an --mode=analysis --model_type=ir50 --imb_factor=0.1 --model_paths checkpoint/$caer_kbcl \
# --model_names KBCL --ckpt_type=best

# python3 analyze_classifier.py --dataset_name=CAER --dataset_path=../data/CAER-S --num_classes=7 \
# --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/caer_ce_an --mode=analysis --model_type=ir50 --imb_factor=0.1 --model_paths checkpoint/$caer_ce \
# --model_names CE --ckpt_type=best



# python3 analyze_classifier.py --dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 \
# --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/af7_compare_kbcl_vs_ce --mode=compare --model_type=kprpe12m --model_paths checkpoint/$af7_ce checkpoint/$af7_kbcl_etf \
# --model_names CE KBCL --ckpt_type=best



python3 analyze_classifier.py --dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 \
--aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
--save_path=results/raf_kbcl_etf_an --mode=analysis --model_type=kprpe12m --model_paths checkpoint/$raf_kbcl_etf \
--model_names KBCL_ETF --ckpt_type=best

# python3 analyze_classifier.py --dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 \
# --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/af7_kbcl_an --mode=analysis --model_type=kprpe12m --model_paths checkpoint/$af7_kbcl \
# --model_names KBCL --ckpt_type=best

# python3 analyze_classifier.py --dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 \
# --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/af7_ce_an --mode=analysis --model_type=kprpe12m --model_paths checkpoint/$af7_ce \
# --model_names CE --ckpt_type=best



# python3 analyze_classifier.py --dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 \
# --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/raf_kbcl_an --mode=analysis --model_type=kprpe12m --model_paths checkpoint/$raf_kbcl \
# --model_names KBCL --ckpt_type=best

# python3 analyze_classifier.py --dataset_name=RAF-DB --dataset_path=../data/RAF-DB_balanced --num_classes=7 \
# --aligner_path=checkpoint/adaface_vit_base_kprpe_webface12m \
# --save_path=results/raf_ce_an --mode=analysis --model_type=kprpe12m --model_paths checkpoint/$raf_ce \
# --model_names CE --ckpt_type=best