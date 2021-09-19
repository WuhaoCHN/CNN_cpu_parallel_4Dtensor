#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <omp.h>
#include "convolution_naive.h"
#include "Relu_naive.h"
#include "pooling_naive.h"
using namespace std;

int main()
{
	cout << "Please enter the size of img data(width, height, channel, batchsize):" << endl;
	double startTime, endTime;
	int width, height, channel, batchsize;
	cin >> width >> height >> channel >> batchsize;
	int pad;
	cout << "Please enter the pad number: " << endl;
	cin >> pad;
	width = width + 2 * pad;
	height = height + 2 * pad;
	int pixels = width * height * channel * batchsize;
	//initialize input data;
	float* img = new float[pixels];
	for(int i = 0 ; i < batchsize;++i)
		for(int j = 0; j < channel;++j)
			for(int k = 0; k < height; ++k)
				for (int l = 0; l < width ; ++l)
				{
					if (l<pad || l>(width - pad - 1) || k < pad || k>(height - pad - 1))
						img[i*channel*width*height + j*width*height + k*width +l] = 0;
					else
						img[i * channel * width * height + j * width * height + k * width + l] = 1;
				}
//	for (int i = 0; i < pixels; ++i)
//		img[i] = 1.0;
//	memset(img , 1.0, sizeof(float) * pixels);
/*	cout << "please enter the input data: " << endl;
	for (int i = 0; i < pixels; ++i)
		cin >> img[i]; */
	//initialize convolutional kernel;
	int ksize, kchannel, knumber;

	kchannel = channel;
	cout << "Please enter the width/height of convolutional kernel: " << endl;
	cin >> ksize;
	cout << "Please enter the number of convolutional kernels: " << endl;
	cin >> knumber;
	int weights = ksize * ksize * kchannel * knumber;
	float* kernel = new float[weights];
	for (int i = 0; i < weights; ++i)
		kernel[i] = 1.0;
//	memset(kernel, 1.0, sizeof(float) * weights);
/*	cout << "please enter the weights of convolutional kernel: " << endl;
	for (int i = 0; i < weights; ++i)
		cin >> kernel[i]; */

	//initialize parameter of convlution and  pooling function
	int stride, PoolingSize;
	float bias;
	cout << "Please enter the stride: " << endl;
	cin >> stride;
	cout << "Please enter the bias: " << endl;
	cin >> bias;
	cout << "Please enter the PoolingSize: " << endl;
	cin >> PoolingSize;

	//convolution and relu output initialization
	int owidth, oheight, ochannel, obatchsize;
	obatchsize = batchsize;
	ochannel = knumber;
	owidth = (width  - ksize) / stride + 1;
	oheight = (height  - ksize) / stride + 1;
	int osumpoints = owidth * oheight * ochannel * obatchsize;
	float* output = new float[osumpoints];
	for (int i = 0; i < osumpoints; ++i)
		output[i] = bias;
	//memset(output, bias, sizeof(float) * osumpoints);


	//pooling output initialization
	int pwidth, pheight, pchannel, pbatchsize;
	pbatchsize = batchsize;
	pchannel = knumber;
	pwidth = owidth / PoolingSize;
	pheight = oheight / PoolingSize;
	int psumpoints = pbatchsize * pchannel * pwidth * pheight;
	float* poutput = new float[psumpoints];
	for (int i = 0; i < psumpoints; ++i)
		poutput[i] = -FLT_MAX;
 //	memset(poutput, -FLT_MAX, sizeof(float) * psumpoints);
	cout << endl;

	startTime = omp_get_wtime();
	convolution(img, channel, height, width, batchsize, ksize, knumber, stride, bias, kernel, output);
	endTime = omp_get_wtime();
	cout << endl;
	cout << "Time consumed by naive convolution : " << endTime - startTime << endl;

	//print convolution output
/*	for (int i = 0; i < batchsize; ++i)
	{
		cout << endl;
		cout << "batch " << i << " :" << endl;
		for (int j = 0; j < knumber; ++j)
		{
			cout << endl;
			cout << "channel " << j << " :" << endl;
			for (int k = 0; k < oheight; ++k)
			{
				cout << endl;
				for (int l = 0; l < owidth; ++l)
					printf("%5.2f\t", output[i * oheight * owidth * knumber + j * oheight * owidth + k * owidth + l]);
			}
		}
	} */
	startTime = omp_get_wtime();
	relu(output, osumpoints);
	endTime = omp_get_wtime();
	cout << endl;
	cout << "Time consumed by naive relu : " << endTime - startTime << endl;


/*	for (int i = 0; i < batchsize; ++i)
	{
		cout << endl;
		cout << "batch " << i << " :" << endl;
		for (int j = 0; j < knumber; ++j)
		{
			cout << endl;
			cout << "channel " << j << " :" << endl;
			for (int k = 0; k < oheight; ++k)
			{
				cout << endl;
				for (int l = 0; l < owidth; ++l)
					printf("%5.2f\t", output[i * oheight * owidth * knumber + j * oheight * owidth + k * owidth + l]);
			}
		}
	}*/
	startTime = omp_get_wtime();
	pooling(output, batchsize, knumber, oheight, owidth, poutput, PoolingSize);
	endTime = omp_get_wtime();
	cout << endl;
	cout << "Time consumed by naive pooling: " << endTime - startTime << endl;
/*	for (int i = 0; i < batchsize; ++i)
	{
		cout << endl;
		cout << "batch " << i << " :" << endl;
		for (int j = 0; j < knumber; ++j)
		{
			cout << endl;
			for (int k = 0; k < pheight; ++k)
			{
				cout << endl;
				for (int l = 0; l<pwidth; ++l)
					printf("%5.2f\t", poutput[i * pheight * pwidth * knumber + j * pheight * pwidth + k * pwidth + l]);
			}
		}
	}*/
	cout << endl;
	delete[]poutput;
	delete[]output;
	delete[]kernel;
	delete []img;
	system("pause");
	return 0;
}