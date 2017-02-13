import theano
import theano.tensor as T
import numpy as np

a = theano.shared(value=np.asarray([1,2,3]), name='a')
m = theano.shared(value=np.zeros((2,2,3)), name='m')

res1 = m + a.dimshuffle('x', 'x', 0)

print res1.eval()
