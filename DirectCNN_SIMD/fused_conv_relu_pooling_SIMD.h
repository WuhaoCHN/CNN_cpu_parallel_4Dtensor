#pragma once
#include <omp.h>

void fused(__m128** bottom, int channel, int height, int width, int batchsize, int ksize, int pnum, int knumber,
	int stride, float bias, __m128* kernel, float** output, float** pout)
{
	//size of output (owidth, oheight, knumber, batchsize)
	int owidth = (width - ksize) / stride + 1;
	int oheight = (height - ksize) / stride + 1;

	//output
	int sum_piece = owidth * oheight;
	int sum_cub = sum_piece * knumber;

	//kernel
	int k_piece = ksize * ksize;

	//pooling
	int pwidth = owidth / pnum;
	int pheight = oheight / pnum;
	int p_piece = pwidth * pheight;

	//img
	int im_piece = height * width;
	int im_cub = im_piece * channel;

	for (int i = 0; i < batchsize; ++i)
		for (int j = 0; j < knumber; ++j)
		{
			for (int k = 0; k < oheight; ++k)
				for (int l = 0; l < owidth; ++l)
				{
					float __declspec(align(32)) op[4] = { 0,0,0,0 };
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
					//relu
					output[i][o_di] = output[i][o_di] > 0 ? output[i][o_di] : 0;
				}
			// pooling
			for (int m = 0; m < pheight; ++m)
				for (int w = 0; w < pwidth; ++w)
					for (int m1 = 0; m1 < pnum; ++m1)
						for (int w1 = 0; w1 < pnum; ++w1)
						{
							int o_d = j * sum_piece + (m * pnum + m1) * owidth + w * pnum + w1;
							int p_d = j * p_piece + m * pwidth + w;
							pout[i][p_d] = pout[i][p_d] > output[i][o_d] ? pout[i][p_d] : output[i][o_d];
						}
		}
}

void fused_cpu(__m128** bottom, int channel, int height, int width, int batchsize, int ksize, int pnum, int knumber,
	int stride, float bias, __m128* kernel, float** output, float** pout)
{
	//size of output (owidth, oheight, knumber, batchsize)
	int owidth = (width - ksize) / stride + 1;
	int oheight = (height - ksize) / stride + 1;

	//output
	int sum_piece = owidth * oheight;
	int sum_cub = sum_piece * knumber;

	//kernel
	int k_piece = ksize * ksize;

	//pooling
	int pwidth = owidth / pnum;
	int pheight = oheight / pnum;
	int p_piece = pwidth * pheight;

	//img
	int im_piece = height * width;
	int im_cub = im_piece * channel;
#pragma omp parallel for
	for (int i = 0; i < batchsize; ++i)
		for (int j = 0; j < knumber; ++j)
		{
			for (int k = 0; k < oheight; ++k)
				for (int l = 0; l < owidth; ++l)
				{
					float __declspec(align(32)) op[4] = { 0,0,0,0 };
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
					//relu
					output[i][o_di] = output[i][o_di] > 0 ? output[i][o_di] : 0;
				}
			// pooling
			for (int m = 0; m < pheight; ++m)
				for (int w = 0; w < pwidth; ++w)
					for (int m1 = 0; m1 < pnum; ++m1)
						for (int w1 = 0; w1 < pnum; ++w1)
						{
							int o_d = j * sum_piece + (m * pnum + m1) * owidth + w * pnum + w1;
							int p_d = j * p_piece + m * pwidth + w;
							pout[i][p_d] = pout[i][p_d] > output[i][o_d] ? pout[i][p_d] : output[i][o_d];
						}
		}
}