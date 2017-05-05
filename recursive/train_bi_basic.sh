mkdir -p log/bi_basic/fig
mkdir -p model/bi_basic
mkdir -p log/summary/bi_basic

# python train_bi_basic.py --fig_path='log/bi_basic/fig/' --bestmodel_dir='model/bi_basic/' --logdir='log/bi_basic/' --train_record='_data/train.record' --valid_record='_data/valid.record' --train_freq=1 --valid_freq=2 --valid_size=2 --lr=0.001 --patience=20 --summary_dir='log/summary/bi_basic/' --max_step=1000000

CUDA_VISIBLE_DEVICES=0 python train_bi_basic.py --fig_path='log/bi_basic/fig/' --bestmodel_dir='model/bi_basic/' --logdir='log/bi_basic/' --train_record='tmp/train.record' --valid_record='tmp/valid.record' --train_freq=10 --valid_freq=200 --valid_size=500 --lr=0.001 --patience=20 --summary_dir='log/summary/bi_basic/' --max_step=1000000 #--load_model=True
