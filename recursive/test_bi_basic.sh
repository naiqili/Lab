mkdir -p log/bi_basic_test/fig

#python test_bi_basic.py --fig_path='log/bi_basic_test/fig/' --bestmodel_dir='model/bi_basic/' --logdir='log/bi_basic_test/' --test_record='tmp/test.record' --test_size=10

CUDA_VISIBLE_DEVICES=0 python test_bi_basic.py --fig_path='log/bi_basic_test/fig/' --bestmodel_dir='model/bi_basic/' --logdir='log/bi_basic_test/' --test_record='tmp/test.record' --test_size=1000
