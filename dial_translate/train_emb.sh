THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32,allow_gc=True,scan.allow_gc=False,nvcc.flags=-use_fast_math python2 train_emb.py
