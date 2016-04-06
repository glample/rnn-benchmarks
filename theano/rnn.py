#!/usr/bin/env python
import time
import optparse
import numpy as np
import theano
import theano.tensor as T


def random_weights(shape):
    drange = np.sqrt(6. / (np.sum(shape)))
    return drange * np.random.uniform(low=-1.0, high=1.0, size=shape)


def create_shared(value, name):
    return theano.shared(value=np.array(value, dtype=np.float32), name=name)


class RNN(object):
    """
    Recurrent neural network. Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (batch_size, sequence_length, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """
    def __init__(self, input_dim, hidden_dim, activation=T.nnet.sigmoid,
                 with_batch=True, name='RNN'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.with_batch = with_batch
        self.name = name

        self.w_x = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_x')
        self.w_h = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_h')

        self.b_h = create_shared(np.zeros((hidden_dim,)), name + '__b_h')
        self.h_0 = create_shared(np.zeros((hidden_dim,)), name + '__h_0')

        self.params = [self.w_x, self.w_h, self.b_h, self.h_0]

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden vector.
        The whole sequence is also accessible through self.h
        """
        def recurrence(x_t, h_tm1):
            return self.activation(x_t + T.dot(h_tm1, self.w_h) + self.b_h)

        # If we used batches, we have to permute the first and second dimension.
        if self.with_batch:
            self.input = input.dimshuffle(1, 0, 2)
            outputs_info = T.alloc(self.h_0, self.input.shape[1], self.hidden_dim)
        else:
            self.input = input
            outputs_info = self.h_0

        h, _ = theano.scan(
            fn=recurrence,
            sequences=T.dot(self.input, self.w_x),
            outputs_info=outputs_info,
            n_steps=self.input.shape[0]
        )
        self.h = h
        self.output = h[-1]

        return self.output


class LSTM(object):
    """
    Long short-term memory (LSTM). Can be used with or without batches.
    Without batches:
        Input: matrix of dimension (sequence_length, input_dim)
        Output: vector of dimension (output_dim)
    With batches:
        Input: tensor3 of dimension (batch_size, sequence_length, input_dim)
        Output: matrix of dimension (batch_size, output_dim)
    """

    def __init__(self, input_dim, hidden_dim, with_batch=True, name='LSTM'):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        # Input gate weights
        self.w_xi = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_xi')
        self.w_hi = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_hi')
        self.w_ci = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_ci')

        # Forget gate weights
        self.w_xf = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_xf')
        self.w_hf = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_hf')
        self.w_cf = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_cf')

        # Output gate weights
        self.w_xo = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_xo')
        self.w_ho = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_ho')
        self.w_co = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_co')

        # Cell weights
        self.w_xc = create_shared(random_weights((input_dim, hidden_dim)), name + '__w_xc')
        self.w_hc = create_shared(random_weights((hidden_dim, hidden_dim)), name + '__w_hc')

        # Initialize the bias vectors, c_0 and h_0 to zero vectors
        self.b_i = create_shared(np.zeros((hidden_dim,)), name + '__b_i')
        self.b_f = create_shared(np.zeros((hidden_dim,)), name + '__b_f')
        self.b_c = create_shared(np.zeros((hidden_dim,)), name + '__b_c')
        self.b_o = create_shared(np.zeros((hidden_dim,)), name + '__b_o')
        self.c_0 = create_shared(np.zeros((hidden_dim,)), name + '__c_0')
        self.h_0 = create_shared(np.zeros((hidden_dim,)), name + '__h_0')

        # Define parameters
        self.params = [self.w_xi, self.w_hi,  # self.w_ci,
                       self.w_xf, self.w_hf,  # self.w_cf,
                       self.w_xo, self.w_ho,  # self.w_co,
                       self.w_xc, self.w_hc,
                       self.b_i, self.b_c, self.b_o, self.b_f,
                      ] # self.c_0, self.h_0]

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden vector.
        The whole sequence is also accessible through self.h
        """

        def recurrence(x_t, c_tm1, h_tm1):
            i_t = T.nnet.sigmoid(T.dot(x_t, self.w_xi) + T.dot(h_tm1, self.w_hi) + self.b_i)  # + T.dot(c_tm1, self.w_ci)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.w_xf) + T.dot(h_tm1, self.w_hf) + self.b_f)  # + T.dot(c_tm1, self.w_cf)
            c_t = f_t * c_tm1 + i_t * T.tanh(T.dot(x_t, self.w_xc) + T.dot(h_tm1, self.w_hc) + self.b_c)
            o_t = T.nnet.sigmoid(T.dot(x_t, self.w_xo) + T.dot(h_tm1, self.w_ho) + self.b_o)  # + T.dot(c_t, self.w_co)
            h_t = o_t * T.tanh(c_t)
            return [c_t, h_t]

        # If we used batches, we have to permute the first and second dimension.
        if self.with_batch:
            self.input = input.dimshuffle(1, 0, 2)
            outputs_info = [T.alloc(x, self.input.shape[1], self.hidden_dim) for x in [self.c_0, self.h_0]]
        else:
            self.input = input
            outputs_info = [self.c_0, self.h_0]

        [c, h], _ = theano.scan(
            fn=recurrence,
            sequences=self.input,
            outputs_info=outputs_info,
            n_steps=self.input.shape[0]
        )
        self.c = c
        self.h = h
        self.output = h[-1]

        return self.output


class FastLSTM(object):
    """
    LSTM with faster implementation (supposedly).
    Not as expressive as the previous one though, because it doesn't include the peepholes connections.
    """
    def __init__(self, input_dim, hidden_dim, with_batch=True, name='LSTM'):
        """
        Initialize neural network.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.with_batch = with_batch
        self.name = name

        self.W = create_shared(random_weights((input_dim, hidden_dim * 4)), name + 'W')
        self.U = create_shared(random_weights((hidden_dim, hidden_dim * 4)), name + 'U')
        self.b = create_shared(random_weights((hidden_dim * 4, )), name + 'b')

        self.c_0 = create_shared(np.zeros((hidden_dim,)), name + '__c_0')
        self.h_0 = create_shared(np.zeros((hidden_dim,)), name + '__h_0')

        self.params = [self.W, self.U, self.b]

    def link(self, input):
        """
        Propagate the input through the network and return the last hidden vector.
        The whole sequence is also accessible through self.h
        """
        def split(x, n, dim):
            return x[:, n*dim:(n+1)*dim]

        def recurrence(x_t, c_tm1, h_tm1):
            p = x_t + T.dot(h_tm1, self.U)
            i = T.nnet.sigmoid(split(p, 0, self.hidden_dim))
            f = T.nnet.sigmoid(split(p, 1, self.hidden_dim))
            o = T.nnet.sigmoid(split(p, 2, self.hidden_dim))
            c = T.tanh(split(p, 3, self.hidden_dim))
            c = f * c_tm1 + i * c
            h = o * T.tanh(c)
            return c, h

        preact = T.dot(input.dimshuffle(1, 0, 2), self.W) + self.b
        outputs_info = [T.alloc(x, input.shape[0], self.hidden_dim) for x in [self.c_0, self.h_0]]

        [_, h], _ = theano.scan(
            fn=recurrence,
            sequences=preact,
            outputs_info=outputs_info,
            n_steps=input.shape[1]
        )
        self.h = h
        self.output = h[-1]

        return self.output


# Parameters

optparser = optparse.OptionParser()
optparser.add_option("-n", "--network_type", default='rnn', help="Network type (rnn, lstm, fastlstm)")
optparser.add_option("-i", "--input_size", default=100, type='int', help="Input layer size")
optparser.add_option("-l", "--hidden_size", default=100, type='int', help="Hidden layer size")
optparser.add_option("-s", "--seq_length", default=30, type='int', help="Sequence length")
optparser.add_option("-b", "--batch_size", default=20, type='int', help="Batch size")
opts = optparser.parse_args()[0]

network_type = opts.network_type
input_size = opts.input_size
hidden_size = opts.hidden_size
seq_length = opts.seq_length
batch_size = opts.batch_size


# Data

n_samples = 100000
x_values = theano.shared(np.random.rand(n_samples, seq_length, input_size).astype(np.float32))
y_values = theano.shared(np.random.rand(n_samples, hidden_size).astype(np.float32))


# Network

start = time.time()

index = T.iscalar()
x = T.ftensor3()
y = T.fmatrix()

if network_type == 'rnn':
    rnn = RNN(input_size, hidden_size)
elif network_type == 'lstm':
    rnn = LSTM(input_size, hidden_size)
elif network_type == 'fastlstm':
    rnn = FastLSTM(input_size, hidden_size)
else:
    raise Exception('Unknown network!')
output = rnn.link(x)

cost = ((output - y) ** 2).mean()
updates = [(p, p - theano.shared(np.float32(0.01)) * g) for p, g in zip(rnn.params, T.grad(cost, rnn.params))]

print 'compiling...'
f_test = theano.function(inputs=[index], outputs=output, givens={x: x_values[index:index + batch_size]})
f_train = theano.function(inputs=[index], outputs=cost, updates=updates, givens={x: x_values[index:index + batch_size], y: y_values[index:index + batch_size]})
f_train(0)
print "Setup : compile + forward/backward x 1"
print "--- %s seconds" % (time.time() - start)

start = time.time()
for k, i in enumerate(xrange(0, n_samples, batch_size)):
    # if k % 100 == 0:
    #     print k
    f_test(i)
end = time.time()
print "Forward:"
print "--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples)

start = time.time()
for k, i in enumerate(xrange(0, n_samples, batch_size)):
    # if k % 100 == 0:
    #     print k
    f_train(i)
end = time.time()
print "Forward + Backward:"
print "--- %i samples in %s seconds (%f samples/s, %.7f s/sample) ---" % (n_samples, end - start, n_samples / (end - start), (end - start) / n_samples)
