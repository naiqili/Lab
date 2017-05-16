mkdir -p log/lstm_binary/fig
mkdir -p model/lstm_binary
mkdir -p log/summary/lstm_binary

#python train_lstm.py --fig_path='log/lstm_binary/fig/' --bestmodel_dir='model/lstm_binary/' --logdir='log/lstm_binary/' --train_record='_data/binary_train.record' --valid_record='_data/binary_valid.record' --train_freq=1 --valid_freq=2 --valid_size=2 --summary_dir='log/summary/lstm_binary/' --L2_lambda=0.0001 --class_size=3 #--load_model=True

CUDA_VISIBLE_DEVICES=0 python train_lstm.py --fig_path='log/lstm_binary/fig/' --bestmodel_dir='model/lstm_binary/' --logdir='log/lstm_binary/' --train_record='_data/binary_train.record' --valid_record='_data/binary_valid.record' --train_freq=10 --valid_freq=200 --valid_size=500 --lr=0.001 --patience=20 --summary_dir='log/summary/lstm_binary/' --max_step=1000000 --L2_lambda=0.0001 --class_size=3 #--load_model=True
