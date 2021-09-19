#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

void convolution(float* bottom, int channel, int height, int width, int batchsize, 
	int ksize, int knumber, int stride, float bias, float* kernel, float* output)
{
	// size of output (owidth, oheight, knumber, batchsize)
	int owidth = (width    - ksize) / stride + 1;
	int oheight = (height  -  ksize) / stride + 1;

	//output
	int sum_piece = owidth * oheight;
	int sum_cub = sum_piece * knumber;
	int osum = sum_cub * batchsize;

	//kernel
	int k_piece = ksize * ksize;
	int k_cub = k_piece * channel;

	//img
	int im_piece = height * width;
	int im_cub = im_piece * channel;




//naive
/*
	for(int i = 0 ; i < batchsize; ++i)
		for(int j = 0; j < knumber; ++j)
			for (int k =0 ; k < oheight; ++k)
				for (int l = 0; l < owidth; ++l)
					for(int m = 0; m < channel; ++m)
						for(int n = 0 ; n < ksize;++n)
							for (int z = 0; z < ksize; ++z)
								output[i * sum_cub + j * sum_piece + k * owidth + l] += kernel[z + n * ksize + m * k_piece + j * k_cub] * bottom[i*im_cub + m*im_piece + (k*stride + n)*width + (l*stride+z)]; */
	for (int i = 0; i < batchsize; ++i)
		for (int j = 0; j < knumber; ++j)
			for (int k = 0; k < oheight; ++k)
				for (int l = 0; l < owidth; ++l)
				{
					int o_di = i * sum_cub + j * sum_piece + k * owidth + l;
					for (int m = 0; m < channel; ++m)
						for (int n = 0; n < ksize; ++n)
							for (int z = 0; z < ksize; ++z)
							{
								int k_di = z + n * ksize + m * k_piece + j * k_cub;
								int b_di = i * im_cub + m * im_piece + (k * stride + n) * width + (l * stride + z);
								output[o_di] += kernel[k_di] * bottom[b_di];
							}
				}
}