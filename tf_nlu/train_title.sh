mkdir -p ./model/title/
mkdir -p ./log/title/
CUDA_VISIBLE_DEVICES=0 python train.py --train_target='title' --modeldir='./model/title/' --logdir='./log/title/' --model_name='title'
