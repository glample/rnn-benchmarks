## Results - Recurrent package
Tested on a Titan-x
```
th recurrent.lua -networkType rnn -hiddenSize 100
Setup : compile + forward/backward x 1
--- 0.036673069000244 seconds ---
Forward:
--- 100000 samples in 7.1649069786072 seconds (13956.907428054 samples/s) ---
Forward + Backward:
--- 100000 samples in 18.566176176071 seconds (5386.1375926401 samples/s) ---


th recurrent.lua -networkType rnn -hiddenSize 500
Setup : compile + forward/backward x 1
--- 0.085377931594849 seconds ---
Forward:
--- 100000 samples in 12.131023168564 seconds (8243.316253702 samples/s) ---
Forward + Backward:
--- 100000 samples in 36.012835979462 seconds (2776.7874612969 samples/s) ---


th recurrent.lua -networkType rnn -hiddenSize 1000
Setup : compile + forward/backward x 1
--- 0.20405006408691 seconds ---

Forward:
--- 100000 samples in 21.969285011292 seconds (4551.8086501038 samples/s) ---
Forward + Backward:
--- 100000 samples in 64.306167125702 seconds (1555.0607355506 samples/s) ---


th recurrent.lua -networkType lstm -hiddenSize 100
Setup : compile + forward/backward x 1
--- 0.34961485862732 seconds ---
Forward:
--- 100000 samples in 26.887173891068 seconds (3719.2429748535 samples/s) ---
Forward + Backward:
--- 100000 samples in 67.390839099884 seconds (1483.881143735 samples/s) ---


th recurrent.lua -networkType lstm -hiddenSize 500
Setup : compile + forward/backward x 1
--- 0.51357793807983 seconds ---
Forward:
--- 100000 samples in 29.653849840164 seconds (3372.2413593724 samples/s) ---
Forward + Backward:
--- 100000 samples in 102.79220104218 seconds (972.83641213161 samples/s) ---
```
