#pragma once
#include <stdio.h>
#include <iostream>
#include <omp.h>
using namespace std;

float im2col_get_pixel(float* im, int height, int width, int channels,
    int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

void im2col(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * height_col * width_col;
    for (c = 0; c < channels_col; ++c) {

        int h_offset = c / (channels * width_col);
        int w_offset = (c / channels) % width_col;
        int c_im = c % channels;


        for (h = 0; h < ksize; ++h) {
            for (w = 0; w < ksize; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * ksize + h) * ksize + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}

void im2col_cpu(float* data_im,
    int channels, int height, int width,
    int ksize, int stride, int pad, float* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;
    int channels_col = channels * height_col * width_col;
#pragma omp parallel for
    for (c = 0; c < channels_col; ++c) {

        int h_offset = c / (channels * width_col);
        int w_offset = (c / channels) % width_col;
        int c_im = c % channels;


        for (h = 0; h < ksize; ++h) {
            for (w = 0; w < ksize; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * ksize + h) * ksize + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}
