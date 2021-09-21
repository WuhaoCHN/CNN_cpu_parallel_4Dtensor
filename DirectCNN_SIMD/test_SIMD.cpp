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
	float** img = new float* [batchsize];
	for (int i = 0; i < batchsize; ++i)
	{
		img[i] = new float[pixels];
		for (int j = 0; j < channel; ++j)
			for(int k = 0 ; k < height ; ++k)
				for (int l = 0; l < width; ++l)
				{
					if (l < pad || width - l - 1 < pad || k < pad || height - k - 1 < pad)
						img[i][j * height * width + k * width + l] = 0;
					else
						img[i][j * height * width + k * width + l] = 0.1;
				}
	} 
//	int channel_re	 = (channel / 4) + bool(channel % 4);
	//channel·½ÏòÉÏpad0 Ê¹µÃchannel ÄÜ±»4Õû³ý
	int channel_re;
	if (channel % 4)
		channel_re = ((channel / 4) + 1) * 4;
	else
		channel_re = channel;
	// img_re[i]: width * height * channel_re;
	int pixels_re = width * height * channel_re;
	float** img_re = new float* [batchsize];
	for (int i = 0; i < batchsize; ++i)
	{
		img_re[i] = new float[pixels_re];
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
		delete[]img[i];
	delete[]img;


	int pix_per_channel_re = channel_re / 4;
	__m128** img_mm = new __m128 * [batchsize];
	for (int i = 0; i < batchsize; ++i)
	{
//		img_mm[i] = new __declspec(align(128)) __m128[width * height];
		img_mm[i] = new __declspec(align(16)) __m128[width * height];


		for (int j = 0; j < width * height; ++j) //channel_re
			for (int k = 0; k < pix_per_channel_re; ++k)
			{
				img_mm[i][j * pix_per_channel_re + k] = _mm_load_ps(&img_re[i][j*channel_re +k*4]);
			}
	} 
	for (int i = 0; i < batchsize; ++i)
		delete[]img_re[i];
	delete[]img_re;


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
	int weights = ksize * ksize * kchannel * knumber;
	float* kernel = new __declspec(align(16)) float[weights];
	for (int i = 0; i < weights; ++i)
		kernel[i] = -1.0;
	float* kernel_re = new __declspec(align(16)) float[ksize * ksize * kchannel_re * knumber];
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
	delete[] kernel;
	__m128* kernel_mm = new __declspec(align(16)) __m128[ksize * ksize * knumber];
	for (int i = 0; i < ksize * ksize * knumber; ++i)
		kernel_mm[i] = _mm_load_ps(&kernel_re[4 * i]);
	delete[] kernel_re;

	int ochannel = knumber;
	int owidth = (width - ksize) / stride + 1;
	int oheight = (height - ksize) / stride + 1;
	float** output = new float* [batchsize];
	for (int i = 0; i < batchsize; ++i)
	{
		output[i] = new __declspec(align(16)) float[owidth * oheight * ochannel];
		for (int j = 0; j < owidth * oheight * ochannel; ++j)
			output[i][j] = bias;
	}


	//convolution
	startTime = omp_get_wtime();
	convolution_cpu(img_mm, channel, height, width, batchsize, ksize, knumber, stride, bias, kernel_mm, output);
	endTime = omp_get_wtime();
	cout << endl;
	cout << "Time consumed by convolution layer using SIMD and openmp: " << endTime - startTime << endl;

	int ochannel_re;
	if (ochannel % 4)
		ochannel_re = ((ochannel / 4) + 1) * 4;
	else
		ochannel_re = ochannel;
	int opixel_re = owidth * oheight * ochannel_re;
	float** output_re = new float* [batchsize];
	for (int i = 0; i < batchsize; ++i)
	{
		output_re[i] = new __declspec(align(16)) float[opixel_re];
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


	__m128** output_mm = new __m128 * [batchsize];
	for (int i = 0; i < batchsize; ++i)
	{
		output_mm[i] = new __declspec(align(16)) __m128[owidth * oheight*pix_per_ochannel_re];
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
	cout << "Time consumed by Relu layer using SIMD and openmp: " << endTime - startTime << endl;
	
	int pwidth = owidth / pnumber;
	int pheight = oheight / pnumber;
	__m128** pout = new __m128 * [batchsize];
	for (int i = 0; i < batchsize; ++i)
	{
		pout[i] = new __declspec(align(16)) __m128[pwidth * pheight * pix_per_ochannel_re];
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
	cout << "Time consumed by Pooling layer using SIMD and Openmp: " << endTime - startTime << endl;

	float** pooling_out = new float* [batchsize];
	for (int i = 0; i < batchsize; ++i)
	{
		pooling_out[i] = new __declspec(align(16)) float[pwidth * pheight * knumber];
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
	cout << "Time consumed by fused_Convolution_Relu_Pooling fuction using SIMD and openmp: " << endTime - startTime << endl;


	for (int i = 0; i < batchsize; ++i)
		delete[]img_mm[i];
	delete[]img_mm;

	for (int i = 0; i < batchsize; ++i)
		delete[]pooling_out[i];
	delete[]pooling_out;
	for (int i = 0; i < batchsize; ++i)
		delete[]output_mm[i];
	delete[]output_mm;                                           
	for (int i = 0; i < batchsize; ++i)
		delete[]output[i];
	delete[]output;
	delete[] kernel_mm;
	system("pause");    
	return 0;
}
