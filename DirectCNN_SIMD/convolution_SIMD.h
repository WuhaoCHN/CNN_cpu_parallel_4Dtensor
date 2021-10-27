#pragma once
#include <omp.h>


void convolution(__m128** bottom, int channel, int height, int width, int batchsize, int ksize, int knumber,
	int stride, float bias, __m128* kernel, float** output)
{
	//size of output (owidth, oheight, knumber, batchsize)
	int owidth = (width - ksize) / stride + 1;
	int oheight = (height - ksize) / stride + 1;

	//output
	int sum_piece = owidth * oheight;
	int sum_cub = sum_piece * knumber;
	
	//kernel
	int k_piece = ksize * ksize;

	//img
	int im_piece = height * width;
	int im_cub = im_piece * channel;
	for(int i = 0; i< batchsize ; ++i)
		for (int j = 0; j < knumber; ++j)
		{

			for (int k = 0; k < oheight; ++k)
				for (int l = 0; l < owidth; ++l)
				{
					__declspec(align(16)) float op[4] = { 0,0,0,0 };
					__m128 temp = _mm_loadu_ps(op);
					int o_di = j * sum_piece + k * owidth + l;
					for (int n = 0; n < ksize; ++n)
						for (int z = 0; z < ksize; ++z)
						{
							int k_di = z + n * ksize + j * k_piece; //
							int b_di = (k * stride + n) * width + l * stride + z;
							temp = _mm_add_ps(temp, _mm_mul_ps(bottom[i][b_di], kernel[k_di]));
						}
					temp = _mm_hadd_ps(temp, temp);
					temp = _mm_hadd_ps(temp, temp);
					_mm_storeu_ps(op, temp);
					output[i][o_di] += op[0];
				}
		}
} 


void convolution_cpu(__m128** bottom, int channel, int height, int width, int batchsize, int ksize, int knumber,
	int stride, float bias, __m128* kernel, float** output)
{
	//size of output (owidth, oheight, knumber, batchsize)
	int owidth = (width - ksize) / stride + 1;
	int oheight = (height - ksize) / stride + 1;

	//output
	int sum_piece = owidth * oheight;
	int sum_cub = sum_piece * knumber;

	//kernel
	int k_piece = ksize * ksize;

	//img
	int im_piece = height * width;
	int im_cub = im_piece * channel;

#pragma omp parallel for num_threads(12)
	for (int i = 0; i < batchsize; ++i)
		for (int j = 0; j < knumber; ++j)
		{

			for (int k = 0; k < oheight; ++k)
				for (int l = 0; l < owidth; ++l)
				{
					__declspec(align(16)) float op[4] = { 0,0,0,0 };
					__m128 temp = _mm_loadu_ps(op);
					int o_di = j * sum_piece + k * owidth + l;
					for (int n = 0; n < ksize; ++n)
						for (int z = 0; z < ksize; ++z)
						{
							int k_di = z + n * ksize + j * k_piece; //
							int b_di = (k * stride + n) * width + l * stride + z;
							temp = _mm_add_ps(temp, _mm_mul_ps(bottom[i][b_di], kernel[k_di]));
						}
					temp = _mm_hadd_ps(temp, temp);
					temp = _mm_hadd_ps(temp, temp);
					_mm_storeu_ps(op, temp);
					output[i][o_di] += op[0];
				}
		}
}
