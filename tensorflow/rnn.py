#!/usr/bin/env python
import time
import optparse
import numpy as np
import tensorflow as tf
# from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn
from six.moves import xrange

def get_feed_dict(x_data, y_data=None):
    feed_dict = {}

    if y_data is not None:
        feed_dict[y] = y_data

    for i in xrange(x_data.shape[0]):
        feed_dict[x[i]] = x_data[i, :, :]

    return feed_dict


# Parameters
optparser = optparse.OptionParser()
optparser.add_option("-n", "--network_type", default='rnn', help="Network type (rnn, lstm, basic_lstm)")
optparser.add_option("-l", "--hidden_size", default=100, type='int', help="Hidden layer size")
optparser.add_option("-s", "--seq_length", default=30, type='int', help="Sequence length")
optparser.add_option("-b", "--batch_size", default=20, type='int', help="Batch size")
opts = optparser.parse_args()[0]

network_type = opts.network_type
print(network_type)
hidden_size = opts.hidden_size
hidden_size = opts.hidden_size
seq_length = opts.seq_length
batch_size = opts.batch_size

n_batch = 1000
n_samples = batch_size * n_batch 

# Data
xinput = np.random.rand(seq_length, batch_size, hidden_size).astype(np.float32)
ytarget = np.random.rand(batch_size, hidden_size).astype(np.float32)

with tf.device('/cpu:0'):

   x = [tf.placeholder(tf.float32, [batch_size, hidden_size], name="x") for i in range(seq_length)]
   y = tf.placeholder(tf.float32, [batch_size, hidden_size], name="y")

   if network_type == 'rnn':
       cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
   elif network_type == 'lstm':
       cell = tf.nn.rnn_cell.LSTMCell(hidden_size, hidden_size)
   elif network_type == 'basic_lstm':
       cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
   else:
       raise Exception('Unknown network! '+network_type)

   print("Compiling...")
   start = time.time()
   # output, _cell_state = rnn.rnn(cell, x, dtype=tf.float32)
   output, _cell_state = rnn.static_rnn(cell, x, dtype=tf.float32)
   cost = tf.reduce_sum((output[-1] - y) ** 2)

   optim = tf.train.GradientDescentOptimizer(0.01)
   train_op = optim.minimize(cost)

   session = tf.Session()
   # session.run(tf.initialize_all_variables())
   session.run(tf.global_variables_initializer())
   session.run(train_op, feed_dict=get_feed_dict(xinput, ytarget))
   print("Setup : compile + forward/backward x 1")
   print("--- %s seconds" % (time.time() - start))

   start = time.time()
   for i in xrange(0, n_batch):
       session.run(output[-1], feed_dict=get_feed_dict(xinput))
   end = time.time()
   print("Forward:")
   print("--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))

   start = time.time()
   for i in xrange(0, n_batch):
       session.run(train_op, feed_dict=get_feed_dict(xinput, ytarget))
   end = time.time()
   print("Forward + Backward:")
   print("--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples))
