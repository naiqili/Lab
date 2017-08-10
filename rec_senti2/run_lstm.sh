model_dir='model/lstm/'
log_dir='log/lstm/'

mkdir -p $model_dir
mkdir -p $log_dir

CUDA_VISIBLE_DEVICES=0 python trainer.py --bestmodel_dir=$model_dir --log_dir=$log_dir --batch_size=100 --lr=0.05 --embed_lr_factor=0.1 --patience=5 --max_epochs=1000 --num_classes=5 --lstm_num_units=300 --keep_prob=0.75 --model_name='LSTMModel' --load_model=False --save_model=True
