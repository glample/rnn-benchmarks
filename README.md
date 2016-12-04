# rnn-benchmarks

All benchmarks are reported for a host with the following specifications :
   * NVIDIA GeForce GTX TITAN X GPU 
   * Intel(R) Xeon(R) CPU E5-2630L v3 @ 1.80GHz
   * CUDA 7.5, cudnnv5

These benchmarks compare the running time of various recurrent neural networks on different deep-learning libraries.
The networks (RNN or LSTM) take as input a 3D Tensor `batch_size x seq_length x hidden_size`
and output the last hidden state, compute a MSE loss, backpropagate the errors through the network and do a simple update of the parameters (`params = params - lr*gradParams`). 
The sequence length is always set to `30`. 
The `hidden_size` specifies the size of the output and input layer of the networks.

The code of the scripts we ran are available. 
The implementations of each model on the different libraries each use 
the fastest implementations we were able to find. 
If you are aware of faster implementations, please let me know. 
I've reported results on Theano, Torch and TensorFlow so far, but we will try to include many more libraries in the future (including cudnn very soon).

The reported `Train` time is the average time needed to run (forward, backward, and update) a training example (and not a training batch), so the smaller the better.
We also report `Compile` time, which includes symbolic graph optimizations (Theano and TensorFlow compilation), as well as a forward and backward pass (to allocate memory).
While the compilation time isn't really a factor in production, it does increase debugging time, which is why we report it here.

## LSTM

This LSTM implementation used for these benchmarks does not use peephole connections between cell and gates.

### Batch Size 32

#### Hidden Size 128

| Library | Compile (s) | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 7.46 | __289.6__ | __99.1__ |
| Torch  | __0.03__ | 434.4 | 99.9 |
| TensorFlow | 3.91 | 820.0 | 266.7 |


#### Hidden Size 512

| Library | Compile (s) | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 7.59 | 619.4 | __200.9__ |
| Torch  | __0.19__ | __610.7__ | 201.7 |
| TensorFlow | 3.97 | 886.9 | 324.9 |


#### Hidden Size 1024

| Library | Compile (s) | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 9.62 | __1013.5__ | __324.1__ |
| Torch  | __0.69__ | 1139.8 | 346.3 |
| TensorFlow | 3.81 | 1329.2 | 562.7 |


### Batch Size 128

#### Hidden Size 128

| Library | Compile (s) | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 7.38 | __102.9__ | 25.6 |
| Torch  | __0.03__ | 109.8 | __25.2__ |
| TensorFlow | 3.68 | 188.6 | 65.0 |


#### Hidden Size 512

| Library | Compile (s) | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 7.50 | 256.0 | 62.8 |
| Torch  | __0.20__ | __214.3__ | __51.4__ |
| TensorFlow | 3.73 | 255.2 | 114.2 |

#### Hidden Size 1024

| Library | Compile (s) | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 7.45 | 583.4 | 160.2 |
| Torch  | __0.75__ | __558.1__ | __112.4__ |
| TensorFlow | 3.84 | 592.2 | 238.1 |


## RNN

This section benchmarks a simple RNN implementation.

### Batch Size 32

#### Hidden Size 128

| Library | Compile (s) | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 4.31 | __104.6__ | __30.9__ |
| Torch  | __0.05__ | 259.53 | 103.06 |
| TensorFlow | 1.64 | 278.4 | 111.5 |

#### Hidden Size 512

| Library | Compile (s) | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 4.36 | __275.2__ | __102.2__ |
| Torch  | __0.05__ | 288.2 | 114.6 |
| TensorFlow | 1.62 | 349.7 | 218.4 |

#### Hidden Size 1024

| Library | Compile (s) | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 4.44 | 443.8 | 179.5 |
| Torch  | __0.09__ | __381.4__ | __118.8__ |
| TensorFlow | 1.72 | 530.0 | 241.7 |

### Batch Size 128

#### Hidden Size 128

| Library | Compile (s) | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 4.48 | __45.4__ | __13.7__ |
| Torch  | __0.08__ | 67.7 | 32.7 |
| TensorFlow | 1.70 | 75.5 | 33.6 |

#### Hidden Size 512

| Library | Compile (s) | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 4.40 | 79.0 | __23.8__ |
| Torch  | __0.09__ | __73.5__ | 34.2 |
| TensorFlow | 1.63 | 125.6 | 86.8 |

#### Hidden Size 1024

| Library | Compile (s) | Train (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- | ------------- |
| Theano | 4.38 | __147.8__ | __50.3__ |
| Torch  | __0.13__ | 150.2 | 64.7 |
| TensorFlow | 1.70 | 222.5 | 137.8 |
