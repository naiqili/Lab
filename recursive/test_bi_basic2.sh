mkdir -p log/bi_basic2_test/fig

#python test_bi_basic2.py --fig_path='log/bi_basic2_test/fig/' --bestmodel_dir='model/bi_basic2/' --logdir='log/bi_basic2_test/' --test_record='tmp/test.record' --test_size=10

CUDA_VISIBLE_DEVICES=0 python test_bi_basic2.py --fig_path='log/bi_basic2_test/fig/' --bestmodel_dir='model/bi_basic2/' --logdir='log/bi_basic2_test/' --test_record='tmp/test.record' --test_size=1000
