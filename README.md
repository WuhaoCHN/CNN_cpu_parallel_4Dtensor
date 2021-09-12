# Intel Test Updated at 2021.9.12
Forward CNNs is implemented in C++. The input data(width, height, channel, batch-size) and convolution kernels(width, height, channel, kernel number) are 4d tensors.
## Introduction
1. The screenshorts of the experiment results are in `Test_results` folder.
2. Each `.h` file contains the naive method and the parallel method using openMP. The parallel method has the suffix `_cpu`.
3. `Test_time.cpp` is used to test the time complexity.
4. `Test_manual.cpp` is used to check the correctness of the code. 
