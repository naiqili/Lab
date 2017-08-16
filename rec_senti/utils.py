import codecs
import functools
import os
import tempfile
import zipfile

from nltk.tokenize import sexpr
import numpy as np
from six.moves import urllib

import tensorflow as tf

def activation(x):
    #return 1.7159 * tf.tanh(0.6666 * x)
    return tf.tanh(x)

def sigmoid(x):
    #return tf.nn.relu(x)
    return tf.sigmoid(x)

def download_and_unzip(data_dir, url_base, zip_name, *file_names):
    zip_path = os.path.join(data_dir, zip_name)
    url = url_base + zip_name
    out_paths = []
    if not os.path.exists(zip_path):
        print('downloading %s to %s' % (url, zip_path))
        urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as f:
        for file_name in file_names:
            if not os.path.exists(os.path.join(data_dir, file_name)):
                print('extracting %s' % file_name)
                out_paths.append(f.extract(file_name, path=data_dir))      
            else:                
                print('already extracted %s' % file_name)
                out_paths.append(os.path.join(data_dir, file_name))
    return out_paths