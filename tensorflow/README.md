#TensorFlow benchmarks

provided by Maarten Bosma.

I used the build-in rnn libary. ``basic_lstm`` is the Tensorflow equivalent of FastLSTM. 

To install TensorFlow follow [these instructions](https://www.tensorflow.org/versions/r0.7/get_started/os_setup.html#pip-installation).

## Results

These results are produced using TensorFlow 0.7.1, cuda 7.5, cudnnv4, turned off ondemand cpu governor [1], Intel(R) Xeon(R) CPU E5-1650 v3 @ 3.50GHz, Titan X:

```
+ python rnn.py -n rnn -b 20 -i 100 -l 100 -s 30
Forward:
--- 100000 samples in 11.039894104 seconds (9058.057117 samples/s) ---
Forward + Backward:
--- 100000 samples in 25.300686121 seconds (3952.461833 samples/s) ---
+ python rnn.py -n rnn -b 20 -i 100 -l 500 -s 30
Forward:
--- 100000 samples in 19.6222681999 seconds (5096.250552 samples/s) ---
Forward + Backward:
--- 100000 samples in 43.0670762062 seconds (2321.959292 samples/s) ---
+ python rnn.py -n basic_lstm -b 20 -i 100 -l 100 -s 30
Forward:
--- 100000 samples in 25.3170599937 seconds (3949.905568 samples/s) ---
Forward + Backward:
--- 100000 samples in 77.6742260456 seconds (1287.428310 samples/s) ---
+ python rnn.py -n basic_lstm -b 20 -i 100 -l 500 -s 30
Forward:
--- 100000 samples in 36.4037480354 seconds (2746.969825 samples/s) ---
Forward + Backward:
--- 100000 samples in 104.032881021 seconds (961.234534 samples/s) ---
+ python rnn.py -n lstm -b 20 -i 100 -l 100 -s 30
Forward:
--- 100000 samples in 26.2394618988 seconds (3811.053590 samples/s) ---
Forward + Backward:
--- 100000 samples in 81.6460819244 seconds (1224.798498 samples/s) ---
+ python rnn.py -n lstm -b 20 -i 100 -l 500 -s 30
Forward:
--- 100000 samples in 36.3097510338 seconds (2754.080981 samples/s) ---
Forward + Backward:
--- 100000 samples in 104.501612902 seconds (956.923021 samples/s) ---
```

 [1] Turning on performance governor: `sudo bash -c 'for i in ls /sys/devices/system/cpu/*/cpufreq/scaling_governor; do echo 'performance' > $i; done;'`
