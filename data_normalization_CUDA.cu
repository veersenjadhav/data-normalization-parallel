#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <stdlib.h>
#include <chrono>

#define ROWS 36634
#define FEATURES 14
#define DATASET "Dataset.csv"

using namespace std;
using namespace chrono;

__global__
void normalization(float *d_dataset, float *d_min, float *d_max, float *d_normalized)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < ROWS && j < FEATURES)
	{
		d_normalized[i * FEATURES + j] = (d_dataset[i * FEATURES + j] - d_min[j]) / (d_max[j] - d_min[j]);
	}
}

int main()
{
	string line, field;

	float* h_dataset = new float[ROWS * FEATURES];
	float* h_min = new float[FEATURES], *h_max = new float[FEATURES];
	float* h_normalized = new float[ROWS * FEATURES];

	float *d_dataset, *d_min, *d_max, *d_normalized;

	ifstream in(DATASET);

	int value = 0;
	while (getline(in, line))
	{
		stringstream ss(line);

		while (getline(ss, field, ','))
		{
			h_dataset[value] = (float)atof(field.c_str());
			value++;
		}
	}

	auto start = steady_clock::now();

#pragma omp parallel
	{
#pragma omp for
		for (int j = 0; j<FEATURES; j++)
		{
			double temp_min = h_dataset[j];
			for (int i = 1; i<ROWS; i++)
			{
				if (h_dataset[i * FEATURES + j] < temp_min)
				{
					temp_min = h_dataset[i * FEATURES + j];
				}
			}
			h_min[j] = temp_min;
		}

#pragma omp for
		for (int j = 0; j<FEATURES; j++)
		{
			double temp_max = h_dataset[j];
			for (int i = 0; i<ROWS; i++)
			{
				if (h_dataset[i * FEATURES + j] > temp_max)
				{
					temp_max = h_dataset[i * FEATURES + j];
				}
			}
			h_max[j] = temp_max;
		}
	}

	auto end = steady_clock::now();

	cudaMalloc((void**)&d_dataset, ROWS*FEATURES * sizeof(float));
	cudaMalloc((void**)&d_min, FEATURES * sizeof(float));
	cudaMalloc((void**)&d_max, FEATURES * sizeof(float));
	cudaMalloc((void**)&d_normalized, ROWS*FEATURES * sizeof(float));

	cudaMemcpy(d_min, h_min, FEATURES * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_max, h_max, FEATURES * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dataset, h_dataset, ROWS*FEATURES * sizeof(float), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(512, 2);
	dim3 blocksPerGrid(72, 7);

	auto start_CUDA = steady_clock::now();

	normalization<<<blocksPerGrid, threadsPerBlock >>>(d_dataset, d_min, d_max, d_normalized);
	cudaDeviceSynchronize();

	auto end_CUDA = steady_clock::now();

	cudaMemcpy(h_normalized, d_normalized, ROWS*FEATURES * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i<5; i++)
	{
		for (int j = 0; j<FEATURES; j++)
		{
			printf("%.8f \t", h_normalized[i * FEATURES + j]);
		}
		printf("\n");
	}

	cout << "\n Elapsed time in seconds : "
		<< chrono::duration_cast<chrono::microseconds>((end - start) + (end_CUDA - start_CUDA)).count()
		<< " microsec" << endl;

	cudaFree(d_dataset); cudaFree(d_min); cudaFree(d_max); cudaFree(d_normalized);

	delete[] h_dataset, h_min, h_max, h_normalized;

	return 0;
}