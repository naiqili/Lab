mkdir -p log/title/basic/fig
mkdir -p model/title/basic/train
mkdir -p model/title/basic/best

CUDA_VISIBLE_DEVICES=0 python train --reset_after=100, --valid_freq=5 --model_dir='model/title/basic/train' --bestmodel_dir='model/title/basic/best' --logdir='log/title/basic' --fig_path='log/title/basic/fig'
