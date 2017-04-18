mkdir -p ./model/invitee/
mkdir -p ./log/invitee/
CUDA_VISIBLE_DEVICES=0 python train.py --train_target='invitee' --modeldir='./model/invitee/' --logdir='./log/invitee/' --model_name='invitee'
