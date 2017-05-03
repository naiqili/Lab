mkdir -p log/bi_basic2/fig
mkdir -p model/bi_basic2
mkdir -p log/summary/bi_basic2

#python train_bi_basic2.py --fig_path='log/bi_basic2/fig/' --bestmodel_dir='model/bi_basic2/' --logdir='log/bi_basic2/' --train_record='_data/train.record' --valid_record='_data/valid.record' --train_freq=1 --valid_freq=2 --valid_size=2 --summary_dir='log/summary/bi_basic2/' --load_model=True

CUDA_VISIBLE_DEVICES=0 python train_bi_basic2.py --fig_path='log/bi_basic2/fig/' --bestmodel_dir='model/bi_basic2/' --logdir='log/bi_basic2/' --train_record='tmp/train.record' --valid_record='tmp/valid.record' --train_freq=10 --valid_freq=200 --valid_size=500 --lr=0.001 --patience=20 --summary_dir='log/summary/bi_basic2/' --load_model=True
