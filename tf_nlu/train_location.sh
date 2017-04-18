mkdir -p ./model/location/
mkdir -p ./log/location/
CUDA_VISIBLE_DEVICES=0 python train.py --train_target='location' --modeldir='./model/location/' --logdir='./log/location/' --model_name='location'
