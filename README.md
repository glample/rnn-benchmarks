# rnn-benchmarks

All benchmarks are reported using a Nvidia GeForce GTX TITAN X GPU.

These benchmarks compare the running time of various recurrent neural networks on different deep-learning libraries.
The networks (RNN or LSTM) take as input a 3D Tensor (batch_size x seq_length x input_size) and output the last hidden state, compute a MSE loss and backpropagate the errors through the network. Input layer size is always set to 100, and sequence length to 30.

The code of the scripts I ran is available. The implementations of each model on the different libraries each use the fastest implementations I was able to find. If you are aware of faster implementations, please let me know. I've reported results on Theano and Torch so far, but I will try to include many more libraries in the future.

The reported time is the average time needed to run a training example (and not a training batch), so the smaller the better.

### RNN

#### Hidden layer size 100 - Batch size 20

| Library | Time (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano  | 253.9 | 87.82 |
| Torch | 315.4 | 121.8 |


#### Hidden layer size 500 - Batch size 20

| Library | Time (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Torch | 376.0 | 143.1 |
| Theano | 498.4 | 182.9 |


#### Hidden layer size 1000 - Batch size 20

| Library | Time (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Torch | 637.4 | 230.2 |
| Theano | 758.8 | 326.3 |



### LSTM

#### Hidden layer size 100 - Batch size 20

| Library | Time (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano (FastLSTM) | 587.7 | 215.1 |
| Theano (LSTM) | 725.3 | 237.5 |
| Torch (Element-Research FastLSTM) | 1017.4 | 367.3 |
| Torch (Element-Research LSTM) | 3549.5 | 1630.8 |


#### Hidden layer size 500 - Batch size 20

| Library | Time (µs) | Forward only (µs) |
| ------------- | ------------- | ------------- |
| Theano (FastLSTM) | 1045.4 | 342.7 |
| Torch (Element-Research FastLSTM) | 1106.5 | 425.2 |
| Theano (LSTM) | 2298.1 | 736.4 |
| Torch (Element-Research LSTM) | 4636.1 | 2923.9 |


FastLSTM implementations (for both Torch and Theano) do not use peephole connections between cell and gates, and compute the input, forget and output gates, as well as the hidden state, in the same operation.
