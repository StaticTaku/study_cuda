#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <memory>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
// We only use 2D grid and 2D block size
struct Matrix 
{
    Matrix(int _width, int _height, bool _onGpu = false):width(_width), height(_height), size(_width*_height), size_byte(_width*_height*sizeof(float)), onGpu(_onGpu)
    {
        if(_onGpu)
            cudaMalloc(&elements, size*sizeof(float));
        else
            elements = new float[size];
    }

    int width;
    int height;
    int size;
    int size_byte;
    bool onGpu;
    float* elements = nullptr;

    Matrix& operator= (const Matrix& MatA)
    {
        if(width == MatA.width && height == MatA.height)
        {
            //Mat(onCpu) = MatA(onCpu)
            if(!onGpu && !MatA.onGpu)
            {
                memcpy(elements, MatA.elements, size_byte);
            //Mat(onGpu) = MatA(onCpu), which means you want to send data from cpu to gpu
            }else if(onGpu && !MatA.onGpu)
            {
                cudaMemcpy(elements, MatA.elements, size_byte, cudaMemcpyHostToDevice);
            //Mat(onCpu) = MatA(onGpu), which means you want to send data from gpu to cpu
            }else if(!onGpu && MatA.onGpu)
            {
                cudaMemcpy(elements, MatA.elements, size_byte, cudaMemcpyDeviceToHost);   
            //Mat(onGpu) = MatA(onGpu)
            }else if(onGpu && MatA.onGpu)
            {
                cudaMemcpy(elements, MatA.elements, size_byte, cudaMemcpyDeviceToDevice);   
            }

            return *this;
        }
        else
        {
            std::cout << "Error: Matrix& operator= (const Matrix& MatA) \n";
            std::exit(1);
        }
    }

    ~Matrix()
    {
        if(elements != nullptr)
        {
            if(onGpu)
            {
                cudaFree(elements);
            }else
            {
                delete[] elements;
            }
        }
    }

};

// Device code
__global__ 
void MatAdd(const Matrix& MatA, const Matrix& MatB, Matrix& MatC)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //Note that (MatA.width, MatA.height) == (MatB.width, MatB.height)
    int id  = row*MatA.width + col;

    if (id < MatA.size)
        MatC.elements[id] = MatA.elements[id] + MatB.elements[id];
    
}



int main() 
{
    //we only use GPU 0
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    int maxThreadsPerSM    = deviceProp.maxThreadsPerMultiProcessor;
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int numberOfSM         = deviceProp.multiProcessorCount;
    int warp_size          = deviceProp.warpSize;

    printf("maxThreadsPerSM: %d\nmaxThreadsPerBlock: %d\nnumberOfSM: %d\nwarp_size: %d\n", 
    maxThreadsPerSM, maxThreadsPerBlock, numberOfSM, warp_size);

    //////////////////////////////////////////////////////////////////////////////////////////
    constexpr int width  = 1000;
    constexpr int height = 1000;

    Matrix h_MatA(width, height);
    Matrix h_MatB(width, height);
    Matrix h_MatC(width, height);

    Matrix d_MatA(width, height, true);
    Matrix d_MatB(width, height, true);
    Matrix d_MatC(width, height, true);

    for(int row = 0;row < height; ++row)
    {
        for(int column = 0;column < width; ++column)
        {
            h_MatA.elements[row*width + column] = 2;
            h_MatB.elements[row*width + column] = 2;
        }
    }

    h_MatA = d_MatA;
    h_MatB = d_MatB;
}
