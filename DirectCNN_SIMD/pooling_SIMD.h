#pragma once
#include"nmmintrin.h"
#include"xmmintrin.h"
#include <omp.h>

void pooling(__m128** bottom, int width, int height, int batchsize, int pnum,  __m128** pout)
{
	int pwidth = width / pnum;
	int pheight = height / pnum;
	for(int i = 0 ; i < batchsize; ++i)
		for(int j = 0 ; j < pheight; ++j)
			for (int k = 0; k < pwidth; ++k)
				for(int l = 0; l< pnum; ++l)
					for (int n = 0; n < pnum; ++n)
					{
						pout[i][j * pwidth + k] = _mm_max_ps(pout[i][j * pwidth + k], bottom[i][(j * pnum + l) * width + k * pnum + n]);
					}
}


void pooling_cpu(__m128** bottom, int width, int height, int batchsize, int pnum, __m128** pout)
{
	int pwidth = width / pnum;
	int pheight = height / pnum;
#pragma omp parallel for collapse(2)
	for (int i = 0; i < batchsize; ++i)
		for (int j = 0; j < pheight; ++j)
			for (int k = 0; k < pwidth; ++k)
				for (int l = 0; l < pnum; ++l)
					for (int n = 0; n < pnum; ++n)
					{
						pout[i][j * pwidth + k] = _mm_max_ps(pout[i][j * pwidth + k], bottom[i][(j * pnum + l) * width + k * pnum + n]);
					}
}
