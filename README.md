# Intel Test Updated at 2021.9.19
Forward CNNs is implemented in C++. The input data(width, height, channel, batch-size) and convolution kernels(width, height, channel, kernel number) are 4d tensors.
## Introduction
1. The screenshorts of the experiment results are in `Test_results` folder.
2. Each `.h` file contains the plain method and the parallel method using openMP. The parallel method has the suffix `_cpu`.
3. `Test_time.cpp` is used to test the time complexity.
4. `Test_manual.cpp` is used to check the correctness of the code. 
5. File `DirectCNN_SIMD` is direct convolution without using im2col.
6. File `im2col+gemm` is indirect convolution using matrix-matrix multiplication.
7. File `NaiveMethod` is the most basic sliding window algorithm, used as a comparation.
8. `Report.docx` changed on 2021.9.19 .
