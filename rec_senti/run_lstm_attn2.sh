model_dir='model/lstm_attn2_cell300_attn100/'
log_dir='log/lstm_attn2_cell300_attn100/'

load_model=False

if [ $load_model = False ]
then
    rm -rf $log_dir
fi

mkdir -p $fig_dir
mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $summary_dir

CUDA_VISIBLE_DEVICES=0 python joint_trainer.py --bestmodel_dir=$model_dir --logdir=$log_dir --train_record='_data/finegrained_train.record' --valid_record='_data/finegrained_valid.record' --train_freq=50 --valid_freq=500 --valid_size=1101 --lr=0.001 --patience=10 --max_step=5000000 --L2_lambda=0.0 --class_size=5 --fw_cell_size=300 --wv_emb_file='tmp/embeddings.pkl' --wv_dict='_data/dict.pkl' --wv_vocab_size=20726 --drop_embed=True --embed_keep_prob=0.6 --drop_weight=False --weight_keep_prob=0.5 --drop_fw_hs=False --fw_hs_keep_prob=0.75 --drop_fw_cs=False --fw_cs_keep_prob=0.5 --output_keep_prob=0.6 --rec_keep_prob=0.7 --model_name='LSTMAttn2Model' --mask_type='subtree_mask' --attn_size=100 --load_model=$load_model --save_model=True
