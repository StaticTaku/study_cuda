#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <memory>
#include <cassert>
#include <omp.h>
#include <chrono>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
// We only use 2D grid and 2D block size
namespace cuda_Matrix
{
    struct Matrix 
    {
        int width;
        int height;
        int size;
        int size_byte;
        int reference_counter = 0;
        bool onGpu;
        float* elements = nullptr;

        //eg.
        //Matrix MatA(width, height); //called here
        Matrix(int _width, int _height, bool _onGpu = false):width(_width), height(_height), size(_width*_height), size_byte(_width*_height*sizeof(float)), onGpu(_onGpu)
        {
            reference_counter++;
            if(_onGpu)
                cudaMalloc(&elements, size*sizeof(float));
            else
                elements = new float[size];
        }
        
        //eg.
        //Matrix MatA(width, height);
        //Matrix MatB = MatA; //called here
        //also called by function using call by value, but call by reference
        Matrix(const Matrix& Mat)
        {
            width = Mat.width;
            height = Mat.height;
            size = Mat.size;
            size_byte = Mat.size_byte;
            reference_counter = Mat.reference_counter + 1;
            onGpu = Mat.onGpu;
            elements = Mat.elements;
        }

        //ex.
        //Matrix MatA(width, height);
        //Matrix MatB(width, height, true);
        //MatA = MatB; //called here
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

        static void Add(const Matrix& MatA, const Matrix& MatB, Matrix& MatC);
        static void Sub(const Matrix& MatA, const Matrix& MatB, Matrix& MatC);
        static void Mul(const Matrix& MatA, const Matrix& MatB, Matrix& MatC);
        static void Had(const Matrix& MatA, const Matrix& MatB, Matrix& MatC);

        ~Matrix()
        {
            if(reference_counter == 1 && elements != nullptr)
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

    struct Reference_SubMatrix
    {
    private:
        float* elements;
    
    public:
        int width;
        int height;
        int stride;

        __host__ __device__
        Reference_SubMatrix(const Matrix& Mat, int row, int col, int _width, int _height):width(_width), height(_height), stride(Mat.width)
        {
            elements = &Mat.elements[stride * row + col];
        }

        __host__ __device__
        float GetElement(int row, int col)
        {
            return elements[row * stride + col];
        }

        __host__ __device__
        void SetElement(int row, int col, float value)
        {
            elements[row * stride + col] = value;
        }
    };


    namespace Kernel
    {
        __global__
        void Add(const Matrix MatA, const Matrix MatB, Matrix MatC)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            //Note that (MatA.width, MatA.height) == (MatB.width, MatB.height) has to be true
            int id  = row*MatA.width + col;
            if (id < MatA.size)
                MatC.elements[id] = MatA.elements[id] + MatB.elements[id];
        }

        __global__
        void Sub(const Matrix MatA, const Matrix MatB, Matrix MatC)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            //Note that (MatA.width, MatA.height) == (MatB.width, MatB.height) has to be true
            int id  = row*MatA.width + col;
            if (id < MatA.size)
                MatC.elements[id] = MatA.elements[id] - MatB.elements[id];
        }

        __global__
        void Had(const Matrix MatA, const Matrix MatB, Matrix MatC)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            //Note that (MatA.width, MatA.height) == (MatB.width, MatB.height) has to be true
            int id  = row*MatA.width + col;
            if (id < MatA.size)
                MatC.elements[id] = MatA.elements[id] * MatB.elements[id];
        }
        
        __global__
        void Mul(const Matrix MatA, const Matrix MatB, Matrix MatC)
        {
            int blockRow = blockIdx.y;
            int blockCol = blockIdx.x;

            //Sub matrix inside MatC
            Reference_SubMatrix Csub(MatC, blockRow*16, blockCol*16, 16, 16);

            float Cvalue = 0;

            //id of the Sub matrix
            int row = threadIdx.y;
            int col = threadIdx.x;

            for (int m = 0; m < ((MatA.width + 16 - 1)/ 16); ++m)
            {
                Reference_SubMatrix Asub(MatA, blockRow*16, m*16, 16, 16);
                Reference_SubMatrix Bsub(MatB, m*16, blockCol*16, 16, 16);
                
                __shared__ float As[16][16];
                __shared__ float Bs[16][16];

                As[row][col] = Asub.GetElement(row, col);
                Bs[row][col] = Bsub.GetElement(row, col);

                __syncthreads();

                for(int e = 0;e < 16;++e)
                    Cvalue += As[row][e] * Bs[e][col];
    
                __syncthreads();
            }
            
            Csub.SetElement(row, col, Cvalue);
        }

        __global__
        void Mul_slow(const Matrix MatA, const Matrix MatB, Matrix MatC)
        {
            float Cvalue = 0;

            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            int MatA_width = MatA.width;
            int MatB_width = MatB.width;
            
            for(int e = 0; e < MatA_width; ++e)
                Cvalue += MatA.elements[row * MatA_width + e] * MatB.elements[e * MatB_width + col];
            MatC.elements[row * MatC.width + col] = Cvalue;
        }
    }

    namespace
    {
        enum class Op_Binary
        {
            Add,
            Sub,
            Had,
            Mul,
        };

        void BinaryOperation(const Matrix& MatA, const Matrix& MatB, Matrix& MatC, Op_Binary op)
        {
            switch (op)
            {
                case Op_Binary::Add:
                case Op_Binary::Sub:
                case Op_Binary::Had:
                    assert(MatA.width == MatB.width && MatA.height == MatB.height && MatC.width == MatA.height && MatC.width == MatA.width);
                    break;
                case Op_Binary::Mul:
                    assert(MatA.width == MatB.height && MatC.width == MatB.width && MatC.height == MatA.height);
                    break;
            }

            int width = MatC.width;
            int height = MatC.height;

            dim3 dimBlock(16,16);
            dim3 dimGrid ((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.x - 1) / dimBlock.y);
            if(MatA.onGpu && MatC.onGpu && MatC.onGpu)//calculation on GPU
            {
                switch (op) 
                {
                    case Op_Binary::Add:
                        Kernel::Add<<<dimGrid, dimBlock>>>(MatA, MatB, MatC);
                        break;
                    case Op_Binary::Sub:
                        Kernel::Sub<<<dimGrid, dimBlock>>>(MatA, MatB, MatC);
                        break;
                    case Op_Binary::Had:
                        Kernel::Had<<<dimGrid, dimBlock>>>(MatA, MatB, MatC);
                        break;
                    case Op_Binary::Mul:
                        Kernel::Mul<<<dimGrid, dimBlock>>>(MatA, MatB, MatC);
                        break;
                }
                cudaDeviceSynchronize();
            }else if(!MatA.onGpu && !MatC.onGpu && !MatC.onGpu)//calculation on CPU
            {
                switch (op) 
                {
                    case Op_Binary::Add:
                        #pragma omp parallel for
                        for(int id = 0;id<MatA.size;++id)
                            MatC.elements[id] = MatA.elements[id] + MatB.elements[id];
                        break;
                    case Op_Binary::Sub:
                        #pragma omp parallel for
                        for(int id = 0;id<MatA.size;++id)
                            MatC.elements[id] = MatA.elements[id] - MatB.elements[id];
                        break;
                    case Op_Binary::Had:
                        #pragma omp parallel for
                        for(int id = 0;id<MatA.size;++id)
                            MatC.elements[id] = MatA.elements[id] * MatB.elements[id];
                        break;
                    case Op_Binary::Mul:
                        #pragma omp parallel for
                        for(int id = 0;id<MatC.size;++id)
                        {
                            float Cvalue = 0;

                            int row = id/MatC.width;
                            int col = id%MatC.width;

                            int MatA_width = MatA.width;
                            int MatB_width = MatB.width;

                            for(int e = 0;e<MatA_width;++e)
                                Cvalue += MatA.elements[row * MatA_width + e] * MatB.elements[e * MatB_width + col];;
                            MatC.elements[id] = Cvalue;
                        }
                        break;
                }
            }
        }
    };

    void Matrix::Add(const Matrix& MatA, const Matrix& MatB, Matrix& MatC)
    {
        BinaryOperation(MatA, MatB, MatC, Op_Binary::Add);
    }

    void Matrix::Sub(const Matrix& MatA, const Matrix& MatB, Matrix& MatC)
    {
        BinaryOperation(MatA, MatB, MatC, Op_Binary::Sub);
    }

    void Matrix::Had(const Matrix& MatA, const Matrix& MatB, Matrix& MatC)
    {
        BinaryOperation(MatA, MatB, MatC, Op_Binary::Had);
    }

    void Matrix::Mul(const Matrix& MatA, const Matrix& MatB, Matrix& MatC)
    {
        BinaryOperation(MatA, MatB, MatC, Op_Binary::Mul);
    }

};

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
    constexpr int width  = 16*100;
    constexpr int height = 16*100;

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

    //mul test using CPU
    {
        Matrix h_MatA(width, height);
        Matrix h_MatB(width, height);
        Matrix h_MatC(width, height);

        //initialize value
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                h_MatA.elements[row*width + column] = 0.5;
                h_MatB.elements[row*width + column] = 0.5;
            }
        }
        start = std::chrono::high_resolution_clock::now();
        Matrix::Mul(h_MatA, h_MatB, h_MatC);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        
        for(int row = 0;row < height; ++row)
        {
            for(int column = 0;column < width; ++column)
            {
                assert(h_MatC.elements[row*width + column] == height*0.25);
            }
        }
    }
    std::cout << "Matrix Multiplication test using CPU passed\n";
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
