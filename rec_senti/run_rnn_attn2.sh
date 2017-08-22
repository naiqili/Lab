model_dir='model/rnn_attn2_cell400_attn150/'
log_dir='log/rnn_attn2_cell400_attn150/'

load_model=True

if [ $load_model = False ]
then
    rm -rf $log_dir
fi

mkdir -p $fig_dir
mkdir -p $model_dir
mkdir -p $log_dir
mkdir -p $summary_dir

CUDA_VISIBLE_DEVICES=0 python joint_trainer.py --bestmodel_dir=$model_dir --logdir=$log_dir --train_record='_data/finegrained_train.record' --valid_record='_data/finegrained_valid.record' --train_freq=50 --valid_freq=500 --valid_size=1101 --lr=0.00005 --patience=10 --max_step=5000000 --L2_lambda=0.0 --class_size=5 --fw_cell_size=400 --wv_emb_file='tmp/embeddings.pkl' --wv_dict='_data/dict.pkl' --wv_vocab_size=20726 --drop_embed=True --embed_keep_prob=0.5 --drop_weight=False --weight_keep_prob=0.5 --drop_fw_hs=False --fw_hs_keep_prob=0.75 --output_keep_prob=0.6 --rec_keep_prob=1.0 --model_name='RNNAttn2Model' --mask_type='subtree_mask' --attn_size=150 --load_model=$load_model --save_model=True
