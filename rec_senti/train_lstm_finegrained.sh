mkdir -p log/lstm_finegrained/fig
mkdir -p model/lstm_finegrained
mkdir -p log/summary/lstm_finegrained

#python train_lstm.py --fig_path='log/lstm_finegrained/fig/' --bestmodel_dir='model/lstm_finegrained/' --logdir='log/lstm_finegrained/' --train_record='_data/finegrained_train.record' --valid_record='_data/finegrained_valid.record' --train_freq=1 --valid_freq=2 --valid_size=2 --summary_dir='log/summary/lstm_finegrained/' --L2_lambda=0.0001 --class_size=5 #--load_model=True

CUDA_VISIBLE_DEVICES=0 python train_lstm.py --fig_path='log/lstm_finegrained/fig/' --bestmodel_dir='model/lstm_finegrained/' --logdir='log/lstm_finegrained/' --train_record='_data/finegrained_train.record' --valid_record='_data/finegrained_valid.record' --train_freq=10 --valid_freq=200 --valid_size=1101 --lr=0.001 --patience=100 --summary_dir='log/summary/lstm_finegrained/' --max_step=1000000 --L2_lambda=0.0001 --class_size=5 --fw_cell_size=150  #--load_model=True
