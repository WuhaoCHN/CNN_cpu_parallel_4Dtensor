#include"nmmintrin.h"
#include"xmmintrin.h"
#include "immintrin.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "convolution_SIMD.h"
#include "Relu_SIMD.h"
#include "pooling_SIMD.h"
#include "fused_conv_relu_pooling_SIMD.h"
using namespace std;


int main()
{
	cout << "Please enter the size of img data(width, height, channel, batchsize):" << endl;
	double startTime, endTime;
	int width, height, channel, batchsize;
	cin >> width >> height >> channel >> batchsize;
	int pad;
	cout << "Please enter the pading number: " << endl;
	cin >> pad;
	width = width + 2 * pad;
	height = height + 2 * pad;
	int pixels = width * height * channel ;  
	int piece = width * height;
	cout << "Please enter the input data: " << endl;
	//float** img = new float* [batchsize];

	float** img = (float**)_aligned_malloc(sizeof(float*) * batchsize, 16);
	for (int i = 0; i < batchsize; ++i)
	{
//		img[i] = new float[pixels];
		img[i] = (float*)_aligned_malloc(pixels * sizeof(float), 16);
		for (int j = 0; j < channel; ++j)
			for(int k = 0 ; k < height ; ++k)
				for (int l = 0; l < width; ++l)
				{
					if (l < pad || width - l - 1 < pad || k < pad || height - k - 1 < pad)
						img[i][j * height * width + k * width + l] = 0;
					else
						cin >>img[i][j * height * width + k * width + l];
					//	cin >> img[i][j * height * width + k * width + l];
				}
	} 
//	int channel_re	 = (channel / 4) + bool(channel % 4);
	//channel方向上pad0 使得channel 能被4整除
	int channel_re;
	if (channel % 4)
		channel_re = ((channel / 4) + 1) * 4;
	else
		channel_re = channel;
	// img_re[i]: width * height * channel_re;
	int pixels_re = width * height * channel_re;
//	float** img_re = new  float* [batchsize];
	float** img_re = (float**)_aligned_malloc(sizeof(float*) * batchsize,16);
	for (int i = 0; i < batchsize; ++i)
	{
//		img_re[i] = new float[pixels_re];
		img_re[i] = (float*)_aligned_malloc(pixels_re * sizeof(float), 16);
		for (int j = 0; j < height; ++j)
			for (int k = 0; k < width; ++k)
				for (int l = 0; l < channel_re; ++l)
				{
					if (l < channel)
						img_re[i][j * width * channel_re + k * channel_re + l] = img[i][l * width * height + j * width + k];
					else
						img_re[i][j * width * channel_re + k * channel_re + l] = 0;
				}
	}
	for (int i = 0; i < batchsize; ++i)
		_aligned_free(img[i]);
	_aligned_free(img);

	int pix_per_channel_re = channel_re / 4;
//	__m128** img_mm = new __m128 * [batchsize];
	__m128** img_mm = (__m128**)_aligned_malloc(sizeof(__m128*) * batchsize,16);
	for (int i = 0; i < batchsize; ++i)
	{
//		img_mm[i] = new __declspec(align(16)) __m128[width * height];
		img_mm[i] = (__m128*)_aligned_malloc(sizeof(__m128)*(width*height),16);
		for (int j = 0; j < width * height; ++j) //channel_re
			for (int k = 0; k < pix_per_channel_re; ++k)
			{
				img_mm[i][j * pix_per_channel_re + k] = _mm_load_ps(&img_re[i][j*channel_re +k*4]);
			}
	} 
	for (int i = 0; i < batchsize; ++i)
		_aligned_free(img_re[i]);
	_aligned_free(img_re);

	//kernel
	int ksize, kchannel, kchannel_re, knumber, stride;
	float bias;
	int pnumber;
	kchannel = channel;
	kchannel_re = channel_re;
	cout << "Please enter the width/height of convolutional kernel: " << endl;
	cin >> ksize;
	cout << "Please enter the number of convolutional kernels: " << endl;
	cin >> knumber;
	cout << "Please enter the stride: " << endl;
	cin >> stride;
	cout << "Please enter the bias: " << endl;
	cin >> bias;
	cout << "Please enter the pooling number: " << endl;
	cin >> pnumber;
	cout << "Please enter the weights of convolutuonal kernel: " << endl;
	int weights = ksize * ksize * kchannel * knumber;
//	float* kernel = new __declspec(align(16)) float[weights];
	float* kernel = (float*)_aligned_malloc(sizeof(float) * weights,16);
	for (int i = 0; i < weights; ++i)
		//	cin >> kernel[i];
		cin >> kernel[i];
//	float* kernel_re = new __declspec(align(16)) float[ksize * ksize * kchannel_re * knumber];
	float* kernel_re = (float*)_aligned_malloc(sizeof(float) * (ksize * ksize * kchannel_re * knumber), 16);
	for(int i = 0 ; i < knumber; ++i)
		for(int j = 0; j < ksize; ++j)
			for(int k = 0; k < ksize; ++k)
				for (int l = 0; l < kchannel_re; ++l)
				{
					if (l < kchannel)
						kernel_re[i * kchannel_re * ksize * ksize + j * kchannel_re * ksize + k * kchannel_re + l] = kernel[i * kchannel * ksize * ksize + l * ksize * ksize + j * ksize + k];
					else
						kernel_re[i * kchannel_re * ksize * ksize + j * kchannel_re * ksize + k * kchannel_re + l] = 0;
				}
//	delete[] kernel;
	_aligned_free(kernel);

//	__m128* kernel_mm = new __declspec(align(16)) __m128[ksize * ksize * knumber];
	__m128* kernel_mm = (__m128*)_aligned_malloc(sizeof(__m128) * (ksize * ksize * knumber),16);

	for (int i = 0; i < ksize * ksize * knumber; ++i)
		kernel_mm[i] = _mm_load_ps(&kernel_re[4 * i]);
	_aligned_free(kernel_re);

	int ochannel = knumber;
	int owidth = (width - ksize) / stride + 1;
	int oheight = (height - ksize) / stride + 1;
//	float** output = new float* [batchsize]
	float** output = (float**)_aligned_malloc(sizeof(float*) * batchsize,16);
	for (int i = 0; i < batchsize; ++i)
	{
	//	output[i] = new __declspec(align(16)) float[owidth * oheight * ochannel];
		output[i] = (float*)_aligned_malloc(sizeof(float) * (owidth * oheight * ochannel),16);
		for (int j = 0; j < owidth * oheight * ochannel; ++j)
			output[i][j] = bias;
	}
	//convolution
	startTime = omp_get_wtime();
	convolution_cpu(img_mm, channel, height, width, batchsize, ksize, knumber, stride, bias, kernel_mm, output);
	endTime = omp_get_wtime();
	cout << endl;
	cout << "Time consumed by convolution layer using SIMD: " << endTime - startTime << endl;

	cout << endl;
	cout << "The output of convolution layer: " << endl;
	for (int i = 0; i < batchsize; ++i)
	{
		cout << endl;
		cout << "batch " << i << ":" << endl;
		for (int j = 0; j < ochannel; ++j)
		{
			cout << endl;
			cout << "channel " << j << ":" << endl;
			for (int k = 0; k < oheight; ++k)
			{
				cout << endl;
				for (int l = 0; l < owidth; ++l)
					printf("%5.2f\t", output[i][j * oheight * owidth + k * owidth + l]);
			}
		}
	} 
	int ochannel_re;
	if (ochannel % 4)
		ochannel_re = ((ochannel / 4) + 1) * 4;
	else
		ochannel_re = ochannel;
	int opixel_re = owidth * oheight * ochannel_re;
//	float** output_re = new float* [batchsize];
	float** output_re = (float**)_aligned_malloc(sizeof(float*) * batchsize,16);
	for (int i = 0; i < batchsize; ++i)
	{
//		output_re[i] = new __declspec(align(16)) float[opixel_re];
		output_re[i] = (float*)_aligned_malloc(sizeof(float) * opixel_re,16);
		for(int j = 0; j < oheight; ++j)
			for(int k = 0; k < owidth; ++k)
				for (int l = 0; l < ochannel_re; ++l)
				{
					if (l < ochannel)
						output_re[i][j * owidth * ochannel_re + k * ochannel_re + l] = output[i][l * owidth * oheight + j * owidth + k];
					else
						output_re[i][j * owidth * ochannel_re + k * ochannel_re + l] = 0;
				}
			
	}
	int pix_per_ochannel_re = ochannel_re / 4;

//	__m128** output_mm = new __m128 * [batchsize];
	__m128** output_mm = (__m128**)_aligned_malloc(sizeof(__m128*) * batchsize,16);
	for (int i = 0; i < batchsize; ++i)
	{
//		output_mm[i] = new __declspec(align(16)) __m128[owidth * oheight*pix_per_ochannel_re];
		output_mm[i] = (__m128*)_aligned_malloc(sizeof(__m128) * (owidth * oheight * pix_per_ochannel_re),16);
		for (int j = 0; j < oheight * owidth; ++j)
			for (int k = 0; k < pix_per_ochannel_re; ++k)
			{
				output_mm[i][j * pix_per_ochannel_re + k] = _mm_load_ps(&output_re[i][j * ochannel_re + k * 4]);
			}
	}
	startTime = omp_get_wtime();
	Relu_cpu(output_mm, owidth, oheight, batchsize, pix_per_ochannel_re);
	endTime = omp_get_wtime();
	cout << endl;
	cout << "Time consumed by Relu layer using SIMD: " << endTime - startTime << endl;
	
	int pwidth = owidth / pnumber;
	int pheight = oheight / pnumber;
//	__m128** pout = new __m128 * [batchsize];
	__m128** pout = (__m128**)_aligned_malloc(sizeof(__m128*) * batchsize,16);

	for (int i = 0; i < batchsize; ++i)
	{
//		pout[i] = new __declspec(align(16)) __m128[pwidth * pheight * pix_per_ochannel_re];
		pout[i] = (__m128*)_aligned_malloc(sizeof(__m128) * (pwidth * pheight * pix_per_ochannel_re),16);
		for (int j = 0; j < pwidth * pheight * pix_per_ochannel_re; ++j)
		{
			float __declspec(align(16)) am[4] = { 0,0,0,0 };
			pout[i][j] = _mm_load_ps(am);
		}
	}

	startTime = omp_get_wtime();
	pooling_cpu(output_mm, owidth, oheight, batchsize, pnumber, pout); 
	endTime = omp_get_wtime();
	cout << endl;
	cout << "Time consumed by Pooling layer using SIMD: " << endTime - startTime << endl;

	cout << "The output of pooling layer: " << endl;
//	float** p_out = new float* [batchsize *pheight *pwidth* pix_per_ochannel_re];
	float** p_out = (float**)_aligned_malloc(sizeof(float*) * (batchsize * pheight * pwidth * pix_per_ochannel_re),16);

	for (int i = 0; i < batchsize * pheight * pwidth * pix_per_ochannel_re; ++i)
	{
	//	p_out[i] = new __declspec(align(16)) float[4];
		p_out[i] = (float*)_aligned_malloc(sizeof(float) * 4 ,16);
		_mm_store_ps(p_out[i], pout[i / (pwidth * pheight * pix_per_ochannel_re)][i % (pwidth * pheight * pix_per_ochannel_re)]);
	}
	for (int i = 0; i < batchsize; ++i)
	{
		cout << endl;
		cout << "batch " << i << ":" << endl;
		for (int j = 0; j < knumber; ++j)
		{
			cout << endl;
			cout << "channel " << j << ":" << endl;
			for (int k = 0; k < pheight; ++k)
			{
				cout << endl;
				for (int l = 0; l < pwidth; ++l)
					printf("%5.2f\t", p_out[i * pheight * pwidth + k * pwidth + l][j]);
			}
		}
	}

	for (int i = 0; i < batchsize * pheight * pwidth * pix_per_ochannel_re; ++i)
		_aligned_free(p_out[i]);
	_aligned_free(p_out);

//	float** pooling_out = new float* [batchsize];
	float** pooling_out = (float**)_aligned_malloc(sizeof(float*) * batchsize,16);
	for (int i = 0; i < batchsize; ++i)
	{
//		pooling_out[i] = new __declspec(align(16)) float[pwidth * pheight * knumber];
		pooling_out[i] = (float*)_aligned_malloc(sizeof(float) * (pwidth * pheight * knumber),16);

		for (int j = 0; j < pwidth * pheight * knumber; ++j)
			pooling_out[i][j] = 0;
	}
	for (int i = 0; i < batchsize; ++i)
	{
		for (int j = 0; j < owidth * oheight * ochannel; ++j)
			output[i][j] = bias;
	}

	startTime = omp_get_wtime();
	fused_cpu(img_mm, channel, height, width, batchsize, ksize ,pnumber, knumber,stride, bias, kernel_mm, output,pooling_out );
	endTime = omp_get_wtime();
	cout << endl;
	cout << "Time consumed by fused_Convolution_Relu_Pooling fuction using SIMD: " << endTime - startTime << endl;


	for (int i = 0; i < batchsize; ++i)
		_aligned_free(pout[i]);
	_aligned_free(pout);
	for (int i = 0; i < batchsize; ++i)
		_aligned_free(output_re[i]);
	_aligned_free(output_re);
	for (int i = 0; i < batchsize; ++i)
		_aligned_free(img_mm[i]);
	_aligned_free(img_mm);
	for (int i = 0; i < batchsize; ++i)
		_aligned_free(pooling_out[i]);
	_aligned_free(pooling_out);
	for (int i = 0; i < batchsize; ++i)
		_aligned_free(output_mm[i]);
	_aligned_free(output_mm);
	for (int i = 0; i < batchsize; ++i)
		_aligned_free(output[i]);
	_aligned_free(output);
	_aligned_free(kernel_mm);
	system("pause");    
	return 0;
}
