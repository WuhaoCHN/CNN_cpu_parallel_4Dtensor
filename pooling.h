#pragma once
#include<algorithm>
using namespace std;
void pooling(float** bottom, int batchsize, int width, int height, int channel, int pooling_num,
	float **result_p)
{
	int sum = width * height;
	int width_p = width / pooling_num;
	int height_p = height / pooling_num;
	int sum_p = width_p * height_p;
	for(int b = 0 ; b < batchsize; ++b)
		for(int c = 0 ; c< channel ; ++ c)
			for(int row = 0 ; row < height_p; ++row)
				for (int col = 0; col < width_p; ++col)
				{
					for (int i = 0; i < pooling_num; ++i)
						for (int j = 0; j < pooling_num; ++j)
						{
							result_p[b][(c * height_p + row) * width_p + col] = max(result_p[b][(c * height_p + row) * width_p + col], bottom[b][(c * height + row*pooling_num + i) * width + col*pooling_num + j]);
						}
				}
}