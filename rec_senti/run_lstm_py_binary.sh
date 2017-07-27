fig_dir='log/lstm_binary_dropout/fig/'
model_dir='model/lstm_binary_dropout/'
log_dir='log/lstm_binary_dropout/'
summary_dir='log/summary/lstm_binary_dropout/'

mkdir -p $fig_dir
mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $summary_dir

python train_lstm_py.py --fig_path=$fig_dir --bestmodel_dir=$model_dir --logdir=$log_dir --train_record='_data/binary_train.record' --valid_record='_data/binary_valid.record' --train_freq=100 --valid_freq=300 --valid_size=872 --lr=0.0001 --patience=100 --summary_dir=$summary_dir --max_step=5000000 --L2_lambda=0.003 --class_size=3 --fw_cell_size=150 --wv_emb_file='tmp/embeddings.pkl' --wv_dict='_data/dict.pkl' --wv_vocab_size=18619 --drop_embed=True --embed_keep_prob=0.7 --drop_weight=True --weight_keep_prob=0.5 --drop_fw_hs=True --fw_hs_keep_prob=0.5 --drop_fw_cs=True --fw_cs_keep_prob=0.5 --model_name='LSTMModel' --load_model=True
