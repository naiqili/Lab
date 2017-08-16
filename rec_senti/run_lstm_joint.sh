fig_dir='log/lstm_joint/fig/'
model_dir='model/lstm_joint/'
log_dir='log/lstm_joint/'
summary_dir='log/summary/lstm_joint/'

rm -rf $log_dir

mkdir -p $fig_dir
mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $summary_dir

CUDA_VISIBLE_DEVICES=0 python joint_trainer.py --fig_path=$fig_dir --bestmodel_dir=$model_dir --logdir=$log_dir --train_record='_data/finegrained_train.record' --valid_record='_data/finegrained_valid.record' --train_freq=100 --valid_freq=500 --valid_size=1101 --lr=0.01 --patience=100 --summary_dir=$summary_dir --max_step=5000000 --L2_lambda=0.00001 --class_size=5 --fw_cell_size=150 --wv_emb_file='tmp/embeddings.pkl' --wv_dict='_data/dict.pkl' --wv_vocab_size=20726 --drop_embed=False --embed_keep_prob=0.7 --drop_weight=False --weight_keep_prob=0.5 --drop_fw_hs=True --fw_hs_keep_prob=0.75 --drop_fw_cs=False --fw_cs_keep_prob=0.5 --model_name='LSTMModel' --load_model=False --save_model=True
