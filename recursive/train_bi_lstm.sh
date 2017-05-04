mkdir -p log/bi_lstm/fig
mkdir -p model/bi_lstm
mkdir -p log/summary/bi_lstm

#python train_bi_lstm.py --fig_path='log/bi_lstm/fig/' --bestmodel_dir='model/bi_lstm/' --logdir='log/bi_lstm/' --train_record='_data/train.record' --valid_record='_data/valid.record' --train_freq=1 --valid_freq=2 --valid_size=2 --summary_dir='log/summary/bi_lstm/' #--load_model=True

CUDA_VISIBLE_DEVICES=0 python train_bi_lstm.py --fig_path='log/bi_lstm/fig/' --bestmodel_dir='model/bi_lstm/' --logdir='log/bi_lstm/' --train_record='tmp/train.record' --valid_record='tmp/valid.record' --train_freq=10 --valid_freq=200 --valid_size=500 --lr=0.001 --patience=20 --summary_dir='log/summary/bi_lstm/' --max_step=1000000 #--load_model=True
