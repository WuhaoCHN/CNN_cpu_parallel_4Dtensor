#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "im2col.h"
#include "mkl.h"
#include "convolution.h"
#include "Relu.h"
#include "pooling.h"
#include "fused_conv_relu_pooling.h"
using namespace std;

int main()
{
    cout << "Please enter the size of img data(width, height, channel, batchsize):" << endl;
    //Initialize the input image
    double startTime, endTime;
    int width, height, channel, batchsize;
    cin >> width >> height >> channel >> batchsize;
    int pixels = width * height * channel;
    float** img = (float**)mkl_malloc(batchsize * sizeof(float), 4);
  //  cout << "Please enter the image data: " << endl; 

    for (int i = 0; i < batchsize; ++i)
    {
        img[i] = (float*)mkl_malloc(pixels * sizeof(float), 32);
        for (int j = 0; j < pixels; ++j)
            img[i][j] = 0.1;
    }
    //Initialize the convolutional layer parameter:
    int ksize, koutput;
    cout << "Please enter the number of convolutional kernels:" << endl;
    cin >> koutput;
    cout << "Please enter the size of convolutional kernels:" << endl;
    cin >> ksize;
    int weights = ksize * ksize * channel;
    cout << "Please enter padding number:" << endl;
    int pad, stride;
    cin >> pad;
    cout << "Please enter the stride:" << endl;
    cin >>  stride;
    cout << "Please enter the bias:" << endl;
    float bias;
    cin >> bias;
 //   cout << "Please enter the weights of convolutional kernel: " << endl; 
    float* k2col = (float*)mkl_malloc((koutput * weights) * sizeof(float), 32);
    for (int i = 0; i < koutput * weights; ++i)
        k2col[i] = 0.1;

    int width_out, height_out, channel_out;
    width_out = (width + (2 * pad) - ksize) / stride + 1;
    height_out = (height + (2 * pad) - ksize) / stride + 1;
    int sum_out = width_out * height_out;
    float** result = (float**)mkl_malloc(batchsize * sizeof(float), 4);
    for (int i = 0; i < batchsize; ++i)
    {
        result[i] = (float*)mkl_malloc((koutput * sum_out) * sizeof(float), 32);
        for (int j = 0; j < koutput * sum_out; ++j)
            result[i][j] = bias;
    }
    //Initialize the pooling layer parameter:
    cout << "Please enter the pooling_num: " << endl;
    int pooling_num;
    cin >> pooling_num;
    float** feature_p = (float**)mkl_malloc(batchsize * sizeof(float), 4);
    for (int i = 0; i < batchsize; ++i)
    {
        feature_p[i] = (float*)mkl_malloc(((sum_out*koutput) / (pooling_num * pooling_num)) * sizeof(float), 32);
        for (int j = 0; j < ((sum_out*koutput) / (pooling_num * pooling_num)); ++j)
            feature_p[i][j] = -FLT_MAX;
    }
    /*
    fused_conv(img, channel, height, width, batchsize, ksize, koutput, stride, pad, bias, k2col, pooling_num, result, feature_p);
    cout << endl;
    cout << "Convolution layer output :" << endl;
    for (int i = 0; i < batchsize; ++i)
    {
        cout << endl;
        cout << "batch " << i << ":" << endl;
        for (int j = 0; j < koutput; ++j)
        {
            cout << endl;
            cout << "channel " << j << ":" << endl;
            for (int k = 0; k < height_out; ++k)
            {
                cout << endl;
                for (int l = 0; l < width_out; ++l)
                    printf("%5.2f\t", result[i][(j * height_out + k) * width_out + l]);
            }
        }
    }
    cout << endl;
    cout << "fused function output: " << endl;
    for (int i = 0; i < batchsize; ++i)
    {
        cout << endl;
        cout << "batch " << i << ":" << endl;
        for (int j = 0; j < koutput; ++j)
        {
            cout << endl;
            cout << "channel " << j << ":" << endl;
            for (int k = 0; k < height_out / pooling_num; ++k)
            {
                cout << endl;
                for (int l = 0; l < width_out / pooling_num; ++l)
                    printf("%5.2f\t", feature_p[i][(j * (height_out / pooling_num) + k) * (width_out / pooling_num) + l]);
            }
        }
    }
    cout << endl;
    */
    startTime = omp_get_wtime();
    convolution_cpu(img, channel, height, width, batchsize, ksize, koutput, stride, pad, bias, k2col, result);
    endTime = omp_get_wtime();
    cout << endl;
    cout << "Time consumed by convolution layer using OpenMP: " << endTime - startTime << endl;
/*    cout << "Convolution layer output :" << endl;
    for (int i = 0; i < batchsize; ++i)
    {
        cout << endl;
        cout << "batch " << i << ":" << endl;
        for (int j = 0; j < koutput; ++j)
        {
            cout << endl;
            cout << "channel " << j << ":" << endl;
            for (int k = 0; k < height_out; ++k)
            {
                cout << endl;
                for (int l = 0; l < width_out; ++l)
                    printf("%5.2f\t", result[i][(j * height_out + k)*width_out + l]);
            }
        }
    } 
    */
    startTime = omp_get_wtime();
    Relu_cpu(result, batchsize, sum_out * koutput);
    endTime = omp_get_wtime();
    cout << endl;
    cout << "Time consumed by Relu layer using Relu using OpenMP: " << endTime - startTime << endl;
 /*   cout << " Relu layer output:" << endl;
    for (int i = 0; i < batchsize; ++i)
    {
        cout << endl;
        cout << "batch " << i << ":" << endl;
        for (int j = 0; j < koutput; ++j)
        {
            cout << endl;
            cout << "channel " << j << ":" << endl;
            for (int k = 0; k < height_out; ++k)
            {
                cout << endl;
                for (int l = 0; l < width_out; ++l)
                    printf("%5.2f\t", result[i][(j * height_out + k) * width_out + l]);
            }
        }
    } */
    cout << endl;
 //   cout << "Pooling layer output: " << endl;  
    startTime = omp_get_wtime();
    pooling_cpu(result, batchsize, width_out, height_out, koutput, pooling_num, feature_p);
    endTime = omp_get_wtime();
    cout << endl;
    cout << "Time consumed by Pooling layer using OpenMP : " << endTime - startTime << endl; 
 /*   for (int i = 0; i < batchsize; ++i)
    {
        cout << endl;
        cout << "batch " << i << ":" << endl;
        for (int j = 0; j < koutput; ++j)
        {
            cout << endl;
            cout << "channel " << j << ":" << endl;
            for (int k = 0; k < height_out/pooling_num; ++k)
            {
                cout << endl;
                for (int l = 0; l < width_out/pooling_num; ++l)
                    printf("%5.2f\t", feature_p[i][(j * (height_out/pooling_num) + k) * (width_out/pooling_num) + l]);
            }
        }
    } */
    cout << endl; 
    startTime = omp_get_wtime();
    fused_conv_cpu(img, channel, height, width, batchsize, ksize, koutput, stride, pad, bias, k2col, pooling_num, result, feature_p);
    endTime = omp_get_wtime();
    cout << "Time consumed by fused convolution relu pooling layer using OpenMP : " << endTime - startTime << endl;
    cout << endl;

    for (int i = 0; i < batchsize; ++i)
        mkl_free(img[i]);
    mkl_free(k2col);
    for (int i = 0; i < batchsize; ++i)
        mkl_free(result[i]);
    for (int i = 0; i < batchsize; ++i)
        mkl_free(feature_p[i]);
    system("PAUSE");
    return 0;
}




































