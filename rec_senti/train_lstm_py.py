import matplotlib

import tensorflow as tf
import numpy as np
import logging
import cPickle
import os, shutil, random
from os import path

from lstm_model import LSTMModel
from trainer import Trainer
from tfrecord_reader import get_data

matplotlib.use('Agg')

import pylab

flags = tf.flags

flags.DEFINE_integer("wv_vocab_size", 50002, "Vocab size")
flags.DEFINE_integer("embed_size", 300, "Embedding size")
flags.DEFINE_integer("class_size", 3, "3 for binary; 5 for fine grained")
flags.DEFINE_integer("fw_cell_size", 50, "Composition cell size")
flags.DEFINE_integer("train_freq", 10, "Training report frequency")
flags.DEFINE_integer("valid_freq", 100, "Validation report frequency")
flags.DEFINE_integer("valid_size", 1000, "Size of validation data")
flags.DEFINE_integer("max_step", 10000, "Max training step")
flags.DEFINE_integer("patience", 5, "Patience")
flags.DEFINE_string("wv_dict", '../tf_nlu/tmp/dict.pkl', "word vec dict file")
flags.DEFINE_string("train_record", '_data/train.record', "training record file")
flags.DEFINE_string("valid_record", '_data/valid.record', "valid record file")
flags.DEFINE_string("fig_path", './log/fig/', "Path to save the figure")
flags.DEFINE_string("bestmodel_dir", './model/best/', "Path to save the best model")
flags.DEFINE_string("logdir", './log/', "Path to save the log")
flags.DEFINE_string("summary_dir", './summary/', "Path to save the summary")
flags.DEFINE_string("wv_emb_file", '../tf_nlu/tmp/embedding.pkl', "word vec embedding file")
flags.DEFINE_float("lr", 0.01, "Learning rate")
flags.DEFINE_float("L2_lambda", 0.0001, "Lambda of L2 loss")
flags.DEFINE_float("embed_keep_prob", 0.95, "Keep prob of embedding dropout")
flags.DEFINE_float("weight_keep_prob", 0.5, "Keep prob of weight dropout")
flags.DEFINE_float("fw_hs_keep_prob", 0.5, "Keep prob of fw_hs dropout")
flags.DEFINE_float("fw_cs_keep_prob", 0.5, "Keep prob of fw_cs dropout")
flags.DEFINE_boolean("load_model", False, "Whether load the best model")
flags.DEFINE_boolean("drop_embed", False, "Whether drop embeddings")
flags.DEFINE_boolean("drop_weight", False, "Whether drop weights")
flags.DEFINE_boolean("drop_fw_hs", False, "Whether drop forward h states")
flags.DEFINE_boolean("drop_fw_cs", False, "Whether drop forward c states")

FLAGS = flags.FLAGS

config_dict = {'wv_vocab_size': FLAGS.wv_vocab_size, \
               'embed_size': FLAGS.embed_size, \
               'fw_cell_size': FLAGS.fw_cell_size, \
               'lr': FLAGS.lr, \
               'L2_lambda': FLAGS.L2_lambda, \
               'wv_emb_file': FLAGS.wv_emb_file, \
               'class_size': FLAGS.class_size, \
               'drop_embed': FLAGS.drop_embed, \
               'embed_keep_prob': FLAGS.embed_keep_prob, \
               'drop_weight': FLAGS.drop_weight, \
               'weight_keep_prob': FLAGS.weight_keep_prob, \
               'drop_fw_hs': FLAGS.drop_fw_hs, \
               'fw_hs_keep_prob': FLAGS.fw_hs_keep_prob, \
               'drop_fw_cs': FLAGS.drop_fw_cs, \
               'fw_cs_keep_prob': FLAGS.fw_cs_keep_prob
              }

trainer = Trainer(FLAGS, LSTMModel, config_dict)
trainer.run()