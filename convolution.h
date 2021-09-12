#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include "im2col.h"
#include "mkl.h"

void convolution(float** bottom,
    int channels, int height, int width, int batchsize,
    int ksize,int koutput, int stride, int pad, float bias, float* kernel, float** conv_out)
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
        col[i] = (float*)mkl_malloc(col_sum * sizeof(float), 32);
        im2col(bottom[i], channels, height, width, ksize, stride, pad, col[i]);
        float alpha = 1.0;
        float beta = 1.0;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
            koutput, col_height, col_width, alpha, kernel, col_width, col[i], col_width, beta, conv_out[i], col_height);
    }
    mkl_free(col);
}
