#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#define N 1000000


__global__ void add(int *a, int *b, int*c)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N)
	{
		c[tid] = a[tid] + b[tid];
		tid += blockDim.x * gridDim.x;
	}
}

// Device code
// Device code
__global__ void MyKernel(int *array, int arrayCount)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    __syncthreads();
    if (idx < arrayCount) {
        array[idx] *= array[idx];
    }
}

int MyKernel_byte(int block_size)
{
    return block_size*4;
}

// Host code
int launchMyKernel(int arrayCount)
{
    int blockSize;      // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device
                        // launch
    int gridSize;       // The actual grid size needed, based on input
                        // size

    cudaOccupancyMaxPotentialBlockSizeVariableSMem(
        &minGridSize,
        &blockSize,
        (void*)MyKernel,
        MyKernel_byte,
        arrayCount);

	std::cout << blockSize << "," << minGridSize << "," << blockSize*minGridSize << std::endl;

    // If interested, the occupancy can be calculated with
    // cudaOccupancyMaxActiveBlocksPerMultiprocessor

    return 0;
}

int main()
{

	launchMyKernel(64);
}