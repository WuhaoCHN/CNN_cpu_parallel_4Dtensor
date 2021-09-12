#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "im2col.h"
#include "mkl.h"

void fused_conv(float** bottom,
    int channels, int height, int width, int batchsize,
    int ksize, int koutput, int stride, int pad, float bias, float* kernel, int pooling_num, float** conv_out , float** result_p)
{
    int width_out, height_out, channel_out;
    width_out = (width + (2 * pad) - ksize) / stride + 1;
    height_out = (height + (2 * pad) - ksize) / stride + 1;
    int col_width = ksize * ksize * channels;
    int col_height = width_out * height_out;
    int col_sum = col_width * col_height;
    float** col = (float**)mkl_malloc(batchsize * sizeof(float), 4);
    for (int i = 0; i < batchsize; ++i)
    {
        //convolution
        col[i] = (float*)mkl_malloc(col_sum * sizeof(float), 32);
        im2col(bottom[i], channels, height, width, ksize, stride, pad, col[i]);
        float alpha = 1.0;
        float beta = 1.0;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            koutput, col_height, col_width, alpha, kernel, col_width, col[i], col_width, beta, conv_out[i], col_height);
        int width_p = width_out / pooling_num;
        int height_p = height_out / pooling_num;
        int sum_p = width_p * height_p;
        for(int c = 0 ; c < koutput;++c)
            for(int row = 0 ; row < height_p;++row)
                for (int col = 0; col < width_p; ++col)
                {
                    for (int k = 0; k < pooling_num; ++k)
                        for (int j = 0; j < pooling_num; ++j)
                            //pooling
                            result_p[i][(c * height_p + row) * width_p + col] = max(result_p[i][(c * height_p + row) * width_p + col], conv_out[i][(c * height_out + row * pooling_num + k) * width_out + col * pooling_num + j]);
                    //relu
                    result_p[i][(c * height_p + row) * width_p + col] = result_p[i][(c * height_p + row) * width_p + col] > 0 ? result_p[i][(c * height_p + row) * width_p + col] : 0;
                }
        mkl_free(col[i]);
    }
    mkl_free(col);
}


void fused_conv_cpu(float** bottom,
    int channels, int height, int width, int batchsize,
    int ksize, int koutput, int stride, int pad, float bias, float* kernel, int pooling_num, float** conv_out, float** result_p)
{
    int width_out, height_out, channel_out;
    width_out = (width + (2 * pad) - ksize) / stride + 1;
    height_out = (height + (2 * pad) - ksize) / stride + 1;
    int col_width = ksize * ksize * channels;
    int col_height = width_out * height_out;
    int col_sum = col_width * col_height;
    float** col = (float**)mkl_malloc(batchsize * sizeof(float), 4);
#pragma omp parallel for num_threads(12)
    for (int i = 0; i < batchsize; ++i)
    {
        //convolution
        col[i] = (float*)mkl_malloc(col_sum * sizeof(float), 32);
        im2col_cpu(bottom[i], channels, height, width, ksize, stride, pad, col[i]);
        float alpha = 1.0;
        float beta = 1.0;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            koutput, col_height, col_width, alpha, kernel, col_width, col[i], col_width, beta, conv_out[i], col_height);
        int width_p = width_out / pooling_num;
        int height_p = height_out / pooling_num;
        int sum_p = width_p * height_p;
        for (int c = 0; c < koutput; ++c)
            for (int row = 0; row < height_p; ++row)
                for (int col = 0; col < width_p; ++col)
                {
                    for (int k = 0; k < pooling_num; ++k)
                        for (int j = 0; j < pooling_num; ++j)
                            //pooling
                            result_p[i][(c * height_p + row) * width_p + col] = max(result_p[i][(c * height_p + row) * width_p + col], conv_out[i][(c * height_out + row * pooling_num + k) * width_out + col * pooling_num + j]);
                    //relu
                    result_p[i][(c * height_p + row) * width_p + col] = result_p[i][(c * height_p + row) * width_p + col] > 0 ? result_p[i][(c * height_p + row) * width_p + col] : 0;
                }
        mkl_free(col[i]);
    }
    mkl_free(col);
}
