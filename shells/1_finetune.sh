
for i in 1 2 3; do 
python3 FER_CL.py --world_size=1 --num_workers=32 \
--learning_rate=1e-6 --batch_size=144 --n_epochs=30 --weight_decay=1e-4 --optimizer=SAM --scheduler=exp \
--dataset_name=AffectNet --dataset_path=../data/AffectNet7 --num_classes=7 --use_sampler=True --img_size=112 \
--model_type=ir50 \
--loss=CE 
done 
wait 


python3 -c "from utils.pushover import send_message; send_message('FER_CL.py finished')"