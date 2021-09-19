#pragma once
#include <omp.h>

void Relu(float** bottom, int batchsize, int pixel_perbatch)
{
	for (int i = 0; i < batchsize; ++i)
		for (int j = 0; j < pixel_perbatch; ++j)
			bottom[i][j] = bottom[i][j] > 0 ? bottom[i][j] : 0;
}

void Relu_cpu(float** bottom, int batchsize, int pixel_perbatch)
{
#pragma omp parallel for collapse(2)
	for (int i = 0; i < batchsize; ++i)
		for (int j = 0; j < pixel_perbatch; ++j)
			bottom[i][j] = bottom[i][j] > 0 ? bottom[i][j] : 0;
}
