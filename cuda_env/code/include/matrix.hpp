#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>
#include <memory>
#include <cassert>
#include <omp.h>
#include <fstream>
#include <chrono>

constexpr int BLOCK_SIZE = 16;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
// We only use 2D grid and 2D block size
namespace cuda_Matrix
{
    template <class T>
    struct devPtr
    {
        T* device_target;

        devPtr(const T& host_target)
        {
            cudaMalloc((void**)&device_target, sizeof(T));
            cudaMemcpy(device_target, &host_target, sizeof(T), cudaMemcpyHostToDevice);
        }

        T* GetDevPtr() { return device_target; }

        ~devPtr(){ cudaFree(device_target); }
    };  

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
        //Matrix MatA = Matrix(width, height) is the same with Matrix MatA(width, height)
        Matrix(int _width, int _height, bool _onGpu = false):width(_width), height(_height), size(_width*_height), size_byte(_width*_height*sizeof(float)), onGpu(_onGpu)
        {
            reference_counter++;
            if(_onGpu)
            {
                cudaMalloc(&elements, size*sizeof(float));
            }
            else
            {
                elements = new float[size];
            }
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
            reference_counter = 1;
            onGpu = Mat.onGpu;
            if(onGpu)
                cudaMalloc(&elements, size*sizeof(float));
            else
                elements = new float[size];
            
            //Mat(onCpu) = MatA(onCpu)
            if(!onGpu && !Mat.onGpu)
            {
                memcpy(elements, Mat.elements, size_byte);
            //Mat(onGpu) = MatA(onGpu)  
            }else if(onGpu && Mat.onGpu)
            {
                cudaMemcpy(elements, Mat.elements, size_byte, cudaMemcpyDeviceToDevice);   
            }
        }

        //ex.
        //Matrix MatA(width, height);
        //Matrix MatB(width, height, true);
        //MatA = MatB; //called here
        //TODO make size changable
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

        Matrix transpose();

        static void   Add(const Matrix& MatA, const Matrix& MatB, Matrix& MatC);
        static Matrix Add(const Matrix& MatA, const Matrix& MatB);

        static void   Sub(const Matrix& MatA, const Matrix& MatB, Matrix& MatC);
        static Matrix Sub(const Matrix& MatA, const Matrix& MatB);

        static void   Had(const Matrix& MatA, const Matrix& MatB, Matrix& MatC);
        static Matrix Had(const Matrix& MatA, const Matrix& MatB);

        static void   Mul(const Matrix& MatA, const Matrix& MatB, Matrix& MatC);
        static Matrix Mul(const Matrix& MatA, const Matrix& MatB);

        static void Affine(const Matrix& Mat, const Matrix& x, const Matrix& bias, Matrix& MatA);
        static Matrix Affine(const Matrix& Mat, const Matrix& x, const Matrix& bias);

        static Matrix ScalarMul(float scalar, const Matrix& Mat);

        static Matrix ArgMax_par_col(const Matrix& x);

        void change_shape(int _width, int _height)
        {
            width = _width; height = _height; size = _width * _height; size_byte = size * sizeof(float);
        }

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
        void Add(const Matrix* MatA, const Matrix* MatB, Matrix* MatC)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            
            //Note that (MatA.width, MatA.height) == (MatB.width, MatB.height) has to be true
            int id  = row*MatA->width + col;
            if (row < MatA->height && col < MatA->width)
                MatC->elements[id] = MatA->elements[id] + MatB->elements[id];
        }

        __global__
        void Sub(const Matrix* MatA, const Matrix* MatB, Matrix* MatC)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            //Note that (MatA.width, MatA.height) == (MatB.width, MatB.height) has to be true
            int id  = row*MatA->width + col;
            if (row < MatA->height && col < MatA->width)
                MatC->elements[id] = MatA->elements[id] - MatB->elements[id];
        }

        __global__
        void Had(const Matrix* MatA, const Matrix* MatB, Matrix* MatC)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            //Note that (MatA.width, MatA.height) == (MatB.width, MatB.height) has to be true
            int id  = row*MatA->width + col;
            if (row < MatA->height && col < MatA->width)
                MatC->elements[id] = MatA->elements[id] * MatB->elements[id];
        }
        
        //TO DO: this function is correct when MatA and MatB has dimmensions that are multiples of BLOCK_SIZE.
        //To support any dimmensions, we should add padded Matrix.
        __global__
        void Mul(const Matrix* MatA, const Matrix* MatB, Matrix* MatC)
        {
            int blockRow = blockIdx.y;
            int blockCol = blockIdx.x;

            //Sub matrix inside MatC
            Reference_SubMatrix Csub(*MatC, blockRow*BLOCK_SIZE, blockCol*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

            float Cvalue = 0;

            //id of the Sub matrix
            int row = threadIdx.y;
            int col = threadIdx.x;
            
            for (int m = 0; m < ((MatA->width + BLOCK_SIZE - 1)/ BLOCK_SIZE); ++m)
            {
                Reference_SubMatrix Asub(*MatA, blockRow*BLOCK_SIZE, m*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
                Reference_SubMatrix Bsub(*MatB, m*BLOCK_SIZE, blockCol*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
                
                __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
                __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

                As[row][col] = Asub.GetElement(row, col);
                Bs[row][col] = Bsub.GetElement(row, col);

                __syncthreads();

                for(int e = 0;e < BLOCK_SIZE;++e)
                    Cvalue += As[row][e] * Bs[e][col];
    
                __syncthreads();
            }
            
            Csub.SetElement(row, col, Cvalue);
        }

        __global__
        void Affine(const Matrix* Mat, const Matrix* x, const Matrix* bias, Matrix* MatA)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            int MatA_width = Mat->width;
            int MatB_width = x->width;
            
            if(row < Mat->height && col < x->width)
            {
                float Cvalue = 0;
                for(int e = 0; e < MatA_width; ++e)
                {   
                    Cvalue += Mat->elements[row * MatA_width + e] * x->elements[e * MatB_width + col];
                }
                MatA->elements[row * MatA->width + col] = Cvalue + bias->elements[row];
            }
        }

        __global__
        void Mul_slow(const Matrix* MatA, const Matrix* MatB, Matrix* MatC)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;

            int MatA_width = MatA->width;
            int MatB_width = MatB->width;
            
            if(row < MatA->height && col < MatB->width)
            {
                float Cvalue = 0;
                for(int e = 0; e < MatA_width; ++e)
                {   
                    Cvalue += MatA->elements[row * MatA_width + e] * MatB->elements[e * MatB_width + col];
                }
                MatC->elements[row * MatC->width + col] = Cvalue;
            }
        }

        __global__
        void set_transpose(const Matrix* MatA, Matrix* MatB)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            int height = MatB->height;
            int width  = MatB->width;

            if(row < height && col < width)
                MatB->elements[row*width + col] = MatA->elements[col*height + row];
        }

        __global__
        void multiple_scalar(float scalar, const Matrix* Mat, Matrix* MatA)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            int height = Mat->height;
            int width  = Mat->width;

            if(row < height && col < width)
                MatA->elements[row*width + col] = scalar*Mat->elements[row*width + col];
        }

        __global__
        void RelU(const Matrix* x, Matrix* mask, Matrix* output)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            int height = x->height;
            int width  = x->width;
            if(row < height && col < width)
            {
                int id = row*width + col;
                if(x->elements[id] <= 0)
                {
                    output->elements[id] = 0; 
                    mask->elements[id] = 1;
                }else
                {
                    output->elements[id] = x->elements[id]; 
                    mask->elements[id] = 0;
                }
            }
        }

        __global__
        void hide_with_mask(const Matrix* delta, const Matrix* mask, Matrix* output)
        {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            int height = delta->height;
            int width  = delta->width;

            if(row < height && col < width)
            {
                int id = row*width + col;
                if(mask->elements[id] <= 0.5)
                {
                    output->elements[id] = delta->elements[id];
                }else
                {
                    output->elements[id] = 0;
                }
            }
        }

        __global__
        void set_max_id_of_value_par_col(const Matrix* x, Matrix* max_value)
        {
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            int max_id = 0;
            float max = -1e5;

            if(col < x->width)
            {
                max = x->elements[col];
                for(int row = 1;row < x->height;++row)
                {
                    if(x->elements[row * x->width + col] > max)
                    {
                        max_id = row;
                        max = x->elements[row * x->width + col];
                    }
                }
                max_value->elements[col] = max_id;
            }
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
                    assert(MatA.width == MatB.width && MatA.height == MatB.height && MatC.width == MatA.width && MatC.height == MatA.height);
                    break;
                case Op_Binary::Mul:
                    assert(MatA.width == MatB.height && MatC.width == MatB.width && MatC.height == MatA.height);
                    break;
            }

            int width = MatC.width;
            int height = MatC.height;

            dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
            dim3 dimGrid ((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
            //std::cout << width << std::endl;
            //std::cout << dimGrid.x << " " << dimGrid.y << std::endl;
            if(MatA.onGpu && MatC.onGpu && MatC.onGpu)//calculation on GPU
            {
                switch (op) 
                {
                    case Op_Binary::Add:
                        Kernel::Add<<<dimGrid, dimBlock>>>(devPtr<Matrix>(MatA).GetDevPtr(), devPtr<Matrix>(MatB).GetDevPtr(), devPtr<Matrix>(MatC).GetDevPtr());
                        break;
                    case Op_Binary::Sub:
                        Kernel::Sub<<<dimGrid, dimBlock>>>(devPtr<Matrix>(MatA).GetDevPtr(), devPtr<Matrix>(MatB).GetDevPtr(), devPtr<Matrix>(MatC).GetDevPtr());
                        break;
                    case Op_Binary::Had:
                        Kernel::Had<<<dimGrid, dimBlock>>>(devPtr<Matrix>(MatA).GetDevPtr(), devPtr<Matrix>(MatB).GetDevPtr(), devPtr<Matrix>(MatC).GetDevPtr());
                        break;
                    case Op_Binary::Mul:
                        Kernel::Mul_slow<<<dimGrid, dimBlock>>>(devPtr<Matrix>(MatA).GetDevPtr(), devPtr<Matrix>(MatB).GetDevPtr(), devPtr<Matrix>(MatC).GetDevPtr());
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

    Matrix Matrix::transpose()
    {
        Matrix transpose(height, width, onGpu);

        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
        dim3 dimGrid ((transpose.width + dimBlock.x - 1) / dimBlock.x, (transpose.height + dimBlock.y - 1) / dimBlock.y);

        Kernel::set_transpose<<<dimGrid, dimBlock>>>(devPtr<Matrix>(*this).GetDevPtr(), devPtr<Matrix>(transpose).GetDevPtr());
        cudaDeviceSynchronize();
        return transpose;
    }

    void Matrix::Add(const Matrix& MatA, const Matrix& MatB, Matrix& MatC)
    {
        BinaryOperation(MatA, MatB, MatC, Op_Binary::Add);
    }

    Matrix Matrix::Add(const Matrix& MatA, const Matrix& MatB)
    {
        Matrix MatC(MatA.width, MatA.height, MatA.onGpu);
        BinaryOperation(MatA, MatB, MatC, Op_Binary::Add);
        return MatC;
    }

    void Matrix::Sub(const Matrix& MatA, const Matrix& MatB, Matrix& MatC)
    {
        BinaryOperation(MatA, MatB, MatC, Op_Binary::Sub);
    }

    Matrix Matrix::Sub(const Matrix& MatA, const Matrix& MatB)
    {
        Matrix MatC(MatA.width, MatA.height, MatA.onGpu);
        BinaryOperation(MatA, MatB, MatC, Op_Binary::Sub);
        return MatC;
    }

    void Matrix::Had(const Matrix& MatA, const Matrix& MatB, Matrix& MatC)
    {
        BinaryOperation(MatA, MatB, MatC, Op_Binary::Had);
    }

    Matrix Matrix::Had(const Matrix& MatA, const Matrix& MatB)
    {
        Matrix MatC(MatA.width, MatA.height, MatA.onGpu);
        BinaryOperation(MatA, MatB, MatC, Op_Binary::Had);
        return MatC;
    }

    void Matrix::Mul(const Matrix& MatA, const Matrix& MatB, Matrix& MatC)
    {
        BinaryOperation(MatA, MatB, MatC, Op_Binary::Mul);
    }

    Matrix Matrix::Mul(const Matrix& MatA, const Matrix& MatB)
    {
        Matrix MatC(MatB.width, MatA.height, MatA.onGpu);
        BinaryOperation(MatA, MatB, MatC, Op_Binary::Mul);
        return MatC;
    }

    void Matrix::Affine(const Matrix& Mat, const Matrix& x, const Matrix& bias, Matrix& MatA)
    {
        assert(Mat.width == x.height && bias.height == Mat.height);
        assert(bias.width == 1);
        assert(MatA.width == x.width && MatA.height == Mat.height);
        int width = MatA.width;
        int height = MatA.height;

        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
        dim3 dimGrid ((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);

        Kernel::Affine<<< dimGrid, dimBlock >>>(devPtr<Matrix>(Mat).GetDevPtr(), devPtr<Matrix>(x).GetDevPtr(), devPtr<Matrix>(bias).GetDevPtr(), devPtr<Matrix>(MatA).GetDevPtr());
        cudaDeviceSynchronize();
    }

    Matrix Matrix::Affine(const Matrix& Mat, const Matrix& x, const Matrix& bias)
    {
        assert(Mat.width == x.height && bias.height == Mat.height);
        assert(bias.width == 1);
        Matrix MatA(x.width, Mat.height, Mat.onGpu);

        int width = MatA.width;
        int height = MatA.height;
    
        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
        dim3 dimGrid ((width + dimBlock.x - 1) / dimBlock.x, (height + dimBlock.y - 1) / dimBlock.y);
        Kernel::Affine<<< dimGrid, dimBlock>>>(devPtr<Matrix>(Mat).GetDevPtr(), devPtr<Matrix>(x).GetDevPtr(), devPtr<Matrix>(bias).GetDevPtr(), devPtr<Matrix>(MatA).GetDevPtr());
        cudaDeviceSynchronize();
        
        return MatA;
    }

    Matrix Matrix::ScalarMul(float scalar, const Matrix& Mat)
    {
        Matrix output(Mat.width, Mat.height, Mat.onGpu);

        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
        dim3 dimGrid ((output.width + dimBlock.x - 1) / dimBlock.x, (output.height + dimBlock.y - 1) / dimBlock.y);
        Kernel::multiple_scalar<<<dimGrid, dimBlock>>>(scalar, devPtr<Matrix>(Mat).GetDevPtr(), devPtr<Matrix>(output).GetDevPtr());
        cudaDeviceSynchronize();
        return output;
    }

    Matrix Matrix::ArgMax_par_col(const Matrix& x)
    {
        Matrix max_value(x.width, 1, x.onGpu);//max_value[col] = max(x[0][col],x[1][col], , ,)
        dim3 dimBlock(32*6);
        dim3 dimGrid ((x.width + dimBlock.x - 1) / dimBlock.x);
        Kernel::set_max_id_of_value_par_col<<< dimGrid, dimBlock >>>(devPtr<Matrix>(x).GetDevPtr(), devPtr<Matrix>(max_value).GetDevPtr());
        cudaDeviceSynchronize();
        return max_value;
    }

    

    void print_matrix(const Matrix& data)
    {
        if(!data.onGpu)
        {
            for(int row = 0;row<data.height;++row)
                for(int column = 0;column<data.width;++column)
                    printf("%d,%d,%f\n", row, column, data.elements[row*data.width + column]);
        }else
        {
            Matrix copy(data.width, data.height);
            copy = data;
            for(int row = 0;row<data.height;++row)
                for(int column = 0;column<data.width;++column)
                    printf("%d,%d,%f\n", row, column, copy.elements[row*data.width + column]);
        }
    }

    void if_same_matrix(const Matrix& MatA, const Matrix& MatB)
    {
        assert(!MatA.onGpu);
        for(int row = 0;row<MatA.height;++row)
            for(int column = 0;column<MatA.width;++column)
                assert(MatA.elements[row*MatA.width + column] - MatB.elements[row*MatA.width + column] < 1e-5);
    }

    void load_data(std::string file_path, Matrix& data)
    {
        std::ifstream file(file_path);
        assert(file);
        for(int row = 0;row<data.height;++row)
            for(int column = 0;column<data.width;++column)
                file >> data.elements[row*data.width + column];
        file.close();
    }


};