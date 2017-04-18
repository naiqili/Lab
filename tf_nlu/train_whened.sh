mkdir -p ./model/whened/
mkdir -p ./log/whened/
CUDA_VISIBLE_DEVICES=0 python train.py --train_target='whened' --modeldir='./model/whened/' --logdir='./log/whened/' --model_name='whened'
