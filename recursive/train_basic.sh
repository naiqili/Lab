mkdir -p log/basic/fig
mkdir -p model/basic

CUDA_VISIBLE_DEVICES=0 python train_basic.py --fig_path='log/basic/fig/' --bestmodel_dir='model/basic/' --logdir='log/basic' --train_record='tmp/train.record' --valid_record='tmp/valid.record'
