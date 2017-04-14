CUDA_VISIBLE_DEVICES=0 python train.py --cell_type='GRU' --model_path='./model/GRU/' --log_path='./log/GRU/' --vocab_size=52716 --data='real'--batch_size=500 --train_freq=10 --dev_freq=100
