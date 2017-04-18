mkdir -p ./model/day/
mkdir -p ./log/day/
CUDA_VISIBLE_DEVICES=0 python train.py --train_target='day' --modeldir='./model/day/' --logdir='./log/day/' --model_name='day'
