#pragma once
#include"nmmintrin.h"
#include"xmmintrin.h"
#include <omp.h>

void Relu(__m128** bottom, int width,  int height, int batchsize, int pix_per_channel_re)
{
	float __declspec(align(16)) zero[4] = { 0,0,0,0 };
	__m128 zerocp = _mm_loadu_ps(zero);
	for (int i = 0; i < batchsize; ++i)
		for (int j = 0; j < width * height * pix_per_channel_re; ++j)
			bottom[i][j] = _mm_max_ps(bottom[i][j], zerocp);
}

void Relu_cpu(__m128** bottom, int width, int height, int batchsize, int pix_per_channel_re)
{
	float __declspec(align(16)) zero[4] = { 0,0,0,0 };
	__m128 zerocp = _mm_loadu_ps(zero);
#pragma omp parallel for
	for (int i = 0; i < batchsize; ++i)
		for (int j = 0; j < width * height * pix_per_channel_re; ++j)
			bottom[i][j] = _mm_max_ps(bottom[i][j], zerocp);
}