#include <matrix.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <memory>
#include <cassert>
#include <omp.h>
#include <chrono>

using namespace cuda_Matrix;

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
    int maxThreadsforCpu   = omp_get_max_threads();

    printf("maxThreadsPerSM: %d\nmaxThreadsPerBlock: %d\nnumberOfSM: %d\nwarp_size: %d\nmaxThreadsforCpu: %d\n", 
    maxThreadsPerSM, maxThreadsPerBlock, numberOfSM, warp_size, maxThreadsforCpu);

    printf("\n\n\n\n\n");
    //////////////////////////////////////////////////////////////////////////////////////////
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;
    std::chrono::microseconds::rep duration;
    constexpr int width  = BLOCK_SIZE*100;
    constexpr int height = BLOCK_SIZE*100;

    //mul test using GPU
    {
        Matrix h_MatA(width, height);
        Matrix h_MatB(width, height);
        Matrix h_MatC(width, height);

        Matrix d_MatA(width, height, true);
        Matrix d_MatB(width, height, true);
        Matrix d_MatC(width, height, true);

        //initialize value
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                h_MatA.elements[row*width + column] = 0.5;
                h_MatB.elements[row*width + column] = 0.5;
            }
        }

        //transfer data from host to device
        d_MatA = h_MatA;
        d_MatB = h_MatB;

        start = std::chrono::high_resolution_clock::now();
        Matrix::Mul(d_MatA, d_MatB, d_MatC);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        //transfer data from device to host
        h_MatC = d_MatC;
        
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                assert(h_MatC.elements[row*width + column] == height*0.25);
            }
        }
    }
    std::cout << "Matrix Multiplication test using GPU passed\n";
    std::cout << "duration: " << duration << "(ms)\n";
    
    //add test using GPU
    {
        Matrix h_MatA(width, height);
        Matrix h_MatB(width, height);
        Matrix h_MatC(width, height);

        Matrix d_MatA(width, height, true);
        Matrix d_MatB(width, height, true);
        Matrix d_MatC(width, height, true);

        //initialize value
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                h_MatA.elements[row*width + column] = row;
                h_MatB.elements[row*width + column] = column;
            }
        }

        //transfer data from host to device
        d_MatA = h_MatA;
        d_MatB = h_MatB;

        start = std::chrono::high_resolution_clock::now();
        Matrix::Add(d_MatA, d_MatB, d_MatC);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        //transfer data from device to host
        h_MatC = d_MatC;
        
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                assert(h_MatC.elements[row*width + column] == row + column);
            }
        }
    }
    std::cout << "Matrix Addition test using GPU passed\n";
    std::cout << "duration: " << duration << "(ms)\n";

    //add test using CPU
    {
        Matrix h_MatA(width, height);
        Matrix h_MatB(width, height);
        Matrix h_MatC(width, height);

        //initialize value
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                h_MatA.elements[row*width + column] = row;
                h_MatB.elements[row*width + column] = column;
            }
        }
        start = std::chrono::high_resolution_clock::now();
        Matrix::Add(h_MatA, h_MatB, h_MatC);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                assert(h_MatC.elements[row*width + column] == row + column);
            }
        }
    }
    std::cout << "Matrix Addition test using CPU passed\n";
    std::cout << "duration: " << duration << "(ms)\n";

    //sub test using GPU
    {
        Matrix h_MatA(width, height);
        Matrix h_MatB(width, height);
        Matrix h_MatC(width, height);

        Matrix d_MatA(width, height, true);
        Matrix d_MatB(width, height, true);
        Matrix d_MatC(width, height, true);

        //initialize value
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                h_MatA.elements[row*width + column] = row;
                h_MatB.elements[row*width + column] = column;
            }
        }

        //transfer data from host to device
        d_MatA = h_MatA;
        d_MatB = h_MatB;

        start = std::chrono::high_resolution_clock::now();
        Matrix::Sub(d_MatA, d_MatB, d_MatC);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        //transfer data from device to host
        h_MatC = d_MatC;
        
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                assert(h_MatC.elements[row*width + column] == row - column);
            }
        }
    }
    std::cout << "Matrix Subtraction test using GPU passed\n";
    std::cout << "duration: " << duration << "(ms)\n";

    //sub test using CPU
    {
        Matrix h_MatA(width, height);
        Matrix h_MatB(width, height);
        Matrix h_MatC(width, height);

        //initialize value
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                h_MatA.elements[row*width + column] = row;
                h_MatB.elements[row*width + column] = column;
            }
        }
        start = std::chrono::high_resolution_clock::now();
        Matrix::Sub(h_MatA, h_MatB, h_MatC);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                assert(h_MatC.elements[row*width + column] == row - column);
            }
        }
    }
    std::cout << "Matrix Subtraction test using CPU passed\n";
    std::cout << "duration: " << duration << "(ms)\n";
    

    //had test using GPU
    {
        Matrix h_MatA(width, height);
        Matrix h_MatB(width, height);
        Matrix h_MatC(width, height);

        Matrix d_MatA(width, height, true);
        Matrix d_MatB(width, height, true);
        Matrix d_MatC(width, height, true);

        //initialize value
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                h_MatA.elements[row*width + column] = row;
                h_MatB.elements[row*width + column] = column;
            }
        }

        //transfer data from host to device
        d_MatA = h_MatA;
        d_MatB = h_MatB;

        start = std::chrono::high_resolution_clock::now();
        Matrix::Had(d_MatA, d_MatB, d_MatC);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        //transfer data from device to host
        h_MatC = d_MatC;
        
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                assert(h_MatC.elements[row*width + column] == row * column);
            }
        }
    }
    std::cout << "Matrix Hadamard test using GPU passed\n";
    std::cout << "duration: " << duration << "(ms)\n";

    //had test using GPU
    {
        Matrix h_MatA(width, height);
        Matrix h_MatB(width, height);
        Matrix h_MatC(width, height);

        //initialize value
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                h_MatA.elements[row*width + column] = row;
                h_MatB.elements[row*width + column] = column;
            }
        }
        start = std::chrono::high_resolution_clock::now();
        Matrix::Had(h_MatA, h_MatB, h_MatC);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                assert(h_MatC.elements[row*width + column] == row * column);
            }
        }
    }
    std::cout << "Matrix Hadamard product test using CPU passed\n";
    std::cout << "duration: " << duration << "(ms)\n";
    
}