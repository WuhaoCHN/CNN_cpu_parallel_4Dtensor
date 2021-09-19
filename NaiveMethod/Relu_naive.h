#pragma once
#include <omp.h>
void relu(float* bottom, int sum)
{
	for (int i = 0; i < sum; ++i)
		bottom[i] = bottom[i] > 0 ? bottom[i] : 0;
}