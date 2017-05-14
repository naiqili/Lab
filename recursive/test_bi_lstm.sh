mkdir -p log/bi_lstm_test/fig

#python test_bi_lstm.py --fig_path='log/bi_lstm_test/fig/' --bestmodel_dir='model/bi_lstm/' --logdir='log/bi_lstm_test/' --test_record='tmp/test.record' --test_size=10

CUDA_VISIBLE_DEVICES=0 python test_bi_lstm.py --fig_path='log/bi_lstm_test/fig/' --bestmodel_dir='model/bi_lstm/' --logdir='log/bi_lstm_test/' --test_record='tmp/test.record' --test_size=1000
