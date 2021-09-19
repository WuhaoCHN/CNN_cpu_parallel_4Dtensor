#pragma once
#include <omp.h>
void pooling(float* bottom, int batchsize, int channel, int height, int width,
	float* poutput, int pnum)
{
	// poutput: (batchsize, channel, pheight, pwidth)
	int pwidth = width / pnum;
	int pheight = height / pnum;

	//img
	int isum_cub = channel * height * width;
	int isum_piece = height * width;

	//poutput
	int psum_cub = channel * pheight * pwidth;
	int psum_piece = pheight * pwidth;


	for (int i = 0; i < batchsize; ++i)
		for (int j = 0; j < channel; ++j)
			for (int k = 0; k < pheight; ++k)
				for (int l = 0; l < pwidth; ++l)
					for (int n = 0; n < pnum; ++n)
						for (int z = 0; z < pnum; ++z)
							poutput[i * psum_cub + j * psum_piece + k * pwidth + l] = poutput[i * psum_cub + j * psum_piece + k * pwidth + l] > bottom[i * isum_cub + j * isum_piece + (k*pnum + n) * width + l*pnum + z] ? poutput[i * psum_cub + j * psum_piece + k * pwidth + l] : bottom[i * isum_cub + j * isum_piece + (k * pnum + n) * width + l * pnum + z];
}