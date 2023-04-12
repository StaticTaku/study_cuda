#include <matrix.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

using namespace cuda_Matrix;

__global__ 
void generate_normal_weights(Matrix* _weight, int seed)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int id  = row*_weight->width + col;
    curandState_t state;
    curand_init(seed, id, 0, &state);
    if(row < _weight->height && col < _weight->width)
        _weight->elements[id] = 1;
}

__global__ 
void initialize_bias(Matrix* _bias, int seed)
{
    int id  = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < _bias->size)
        _bias->elements[id] = id;
}

__global__ 
void set_one_vector(Matrix* _bias, int seed)
{
    int id  = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < _bias->size)
        _bias->elements[id] = 1;
}

__global__
void set_sum_value_par_col(const Matrix* x, Matrix* max_value, int two_n)
{
    extern __shared__ float sdata[];//blockDim.x * blockDim.y * sizeof(float)
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = threadIdx.y;

    int sid_x = threadIdx.x;
    int sid_y = threadIdx.y;

    if(id_x < x->width)
    {
        sdata[sid_y * blockDim.x + sid_x] = x->elements[id_y * x->width + id_x];
    }
    __syncthreads();

    for(int s=1; s< two_n; s*=2)
    {
        if(id_x < x->width && id_y % (2*s) == 0 && id_y < two_n)
        {
            //if(id_x == 0 && s == 8)
            //    printf("%d,%d, %f, %f,8\n",sid_y, sid_y+s, sdata[sid_y * blockDim.x + sid_x], sdata[(sid_y + s)* blockDim.x + sid_x]);
            sdata[sid_y * blockDim.x + sid_x] += sdata[(sid_y + s)* blockDim.x + sid_x];
            //if(id_x == 0 && s == 1)
            //    printf("%d, %f, 1\n",sid_y, sdata[sid_y * blockDim.x + sid_x]);
            //if(id_x == 0 && s == 4)
            //    printf("%d, %f,2\n",sid_y, sdata[sid_y * blockDim.x + sid_x]);
        }
        __syncthreads();
    }

    if(id_y == 0 && id_x < x->width)
    {
        max_value->elements[id_x] = sdata[sid_x];
        for(int id = two_n; id<x->height; ++id)
            max_value->elements[id_x] += sdata[id * blockDim.x + sid_x];
    }

}

__global__
void set_max_value_par_col(const Matrix* x, Matrix* max_value, int two_n)
{
    extern __shared__ float sdata[];//blockDim.x * blockDim.y * sizeof(float)
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = threadIdx.y;

    int sid_x = threadIdx.x;
    int sid_y = threadIdx.y;

    if(id_x < x->width)
    {
        sdata[sid_y * blockDim.x + sid_x] = x->elements[id_y * x->width + id_x];
    }
    __syncthreads();

    for(int s=1; s< two_n; s*=2)
    {
        if(id_x < x->width && id_y % (2*s) == 0 && id_y < two_n)
        {
            //if(id_x == 0 && s == 8)
            //    printf("%d,%d, %f, %f,8\n",sid_y, sid_y+s, sdata[sid_y * blockDim.x + sid_x], sdata[(sid_y + s)* blockDim.x + sid_x]);
            sdata[sid_y * blockDim.x + sid_x] = sdata[sid_y * blockDim.x + sid_x] > sdata[(sid_y + s)* blockDim.x + sid_x] ? sdata[sid_y * blockDim.x + sid_x]:sdata[(sid_y + s)* blockDim.x + sid_x];
            //if(id_x == 0 && s == 1)
            //    printf("%d, %f, 1\n",sid_y, sdata[sid_y * blockDim.x + sid_x]);
            //if(id_x == 0 && s == 4)
            //    printf("%d, %f,2\n",sid_y, sdata[sid_y * blockDim.x + sid_x]);
        }
        __syncthreads();
    }

    if(id_y == 0 && id_x < x->width)
    {
        max_value->elements[id_x] = sdata[sid_x];
        for(int id = two_n; id<x->height; ++id)
            max_value->elements[id_x] = max_value->elements[id_x] > sdata[id * blockDim.x + sid_x] ? max_value->elements[id_x]:sdata[id * blockDim.x + sid_x];
    }

}

__global__
void kernel_softmax(const Matrix* x, Matrix* sum_value, Matrix* max_value, Matrix* output, int two_n)
{
    extern __shared__ float sdata[];//blockDim.x * blockDim.y * sizeof(float)

    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = threadIdx.y;

    int sid_x = threadIdx.x;
    int sid_y = threadIdx.y;

    if(id_x < x->width)
    {
        //printf("%d,%d, %f,%f,%f,%f\n",id_y, id_x,x->elements[id_y * x->width + id_x], expf(x->elements[id_y * x->width + id_x]), expf(x->elements[id_y * x->width + id_x]-max_value->elements[id_x]), max_value->elements[id_x]);
        sdata[sid_y * blockDim.x + sid_x] = expf(x->elements[id_y * x->width + id_x]-max_value->elements[id_x]);
    }
    __syncthreads();

    for(int s=1; s< two_n; s*=2)
    {
        if(id_x < x->width && id_y % (2*s) == 0 && id_y < two_n)
        {
            //if(id_x == 0 && s == 1)
            //    printf("%d,%d, %f, %f,1\n",sid_y, sid_y+s, sdata[sid_y * blockDim.x + sid_x], sdata[(sid_y + s)* blockDim.x + sid_x]);
            //if(id_x == 0 && s == 8)
            //    printf("%d,%d, %f, %f,8\n",sid_y, sid_y+s, sdata[sid_y * blockDim.x + sid_x], sdata[(sid_y + s)* blockDim.x + sid_x]);
            sdata[sid_y * blockDim.x + sid_x] += sdata[(sid_y + s)* blockDim.x + sid_x];
            //if(id_x == 0 && s == 1)
            //    printf("%d, %f, 1\n",sid_y, sdata[sid_y * blockDim.x + sid_x]);
            //if(id_x == 0 && s == 4)
            //    printf("%d, %f,4\n",sid_y, sdata[sid_y * blockDim.x + sid_x]);
        }
        __syncthreads();
    }

    if(id_y == 0 && id_x < x->width)
    {
        sum_value->elements[id_x] = sdata[sid_x];
        for(int id = two_n; id<x->height; ++id)
        {
            sum_value->elements[id_x] += sdata[id * blockDim.x + sid_x];
            //printf("%f,%f\n",sum_value->elements[id_x], sdata[id * blockDim.x + sid_x]);
        }
    }
    __syncthreads();
    if(id_x < x->width)
    {
        output->elements[id_y * x->width + id_x] = expf(x->elements[id_y * x->width + id_x]-max_value->elements[id_x])/sum_value->elements[id_x];
        //printf("%d, %d, %f, %f, %f\n",id_y, id_x, x->elements[id_y * x->width + id_x]-max_value->elements[id_x], expf(x->elements[id_y * x->width + id_x]-max_value->elements[id_x])/sum_value->elements[id_x], output->elements[id_y * x->width + id_x]);
    }
}

__global__
void kernel_cross_entropy(const Matrix* x, const Matrix* t, Matrix* E, int two_n)
{
    extern __shared__ float sdata[];//blockDim.x * blockDim.y * sizeof(float)
    int id_x = blockIdx.x * blockDim.x + threadIdx.x;
    int id_y = threadIdx.y;

    int sid_x = threadIdx.x;
    int sid_y = threadIdx.y;

    if(id_x < x->width)
    {
        sdata[sid_y * blockDim.x + sid_x] = -t->elements[id_y * x->width + id_x] * logf(x->elements[id_y * x->width + id_x]);
    }
    __syncthreads();

    for(int s=1; s< two_n; s*=2)
    {
        if(id_x < x->width && id_y % (2*s) == 0 && id_y < two_n)
        {
            //if(id_x == 0 && s == 8)
            //    printf("%d,%d, %f, %f,8\n",sid_y, sid_y+s, sdata[sid_y * blockDim.x + sid_x], sdata[(sid_y + s)* blockDim.x + sid_x]);
            sdata[sid_y * blockDim.x + sid_x] += sdata[(sid_y + s)* blockDim.x + sid_x];
            //if(id_x == 0 && s == 1)
            //    printf("%d, %f, 1\n",sid_y, sdata[sid_y * blockDim.x + sid_x]);
            //if(id_x == 0 && s == 4)
            //    printf("%d, %f,2\n",sid_y, sdata[sid_y * blockDim.x + sid_x]);
        }
        __syncthreads();
    }

    if(id_y == 0 && id_x < x->width)
    {
        E->elements[id_x] = sdata[sid_x];
        for(int id = two_n; id<x->height; ++id)
            E->elements[id_x] += sdata[id * blockDim.x + sid_x];
    }
}

struct Affine
{
    Matrix weight;
    Matrix input;
    Matrix bias;
    Matrix delta_weight;
    Matrix delta_bias;
    Matrix one_vector;

    int input_size, output_size, batch_size;
    bool onGpu;
    Affine(int _input_size, int _output_size, int _batch_size, float weight_init = 0.01, bool _onGpu = false):input_size(_input_size),output_size(_output_size),batch_size(_batch_size),onGpu(_onGpu),
    weight(_input_size, _output_size, _onGpu),
    input(_batch_size, _input_size, _onGpu),
    bias(1, _output_size, _onGpu),
    delta_weight(_input_size, _output_size, _onGpu),
    delta_bias(1, _output_size, _onGpu),
    one_vector(1, _batch_size, _onGpu)
    {
        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
        dim3 dimGrid ((_input_size + dimBlock.x - 1) / dimBlock.x, (_output_size + dimBlock.y - 1) / dimBlock.y);

        int seed = 1;
        generate_normal_weights<<< dimBlock, dimGrid >>>(devPtr<Matrix>(weight).GetDevPtr(), seed);

        int threadsPerBlock = 32*4;
        int blocksPerGrid   = (bias.size + threadsPerBlock - 1) / threadsPerBlock;
        initialize_bias<<< threadsPerBlock, blocksPerGrid >>>(devPtr<Matrix>(bias).GetDevPtr(), seed);
        set_one_vector<<< threadsPerBlock, blocksPerGrid >>>(devPtr<Matrix>(one_vector).GetDevPtr(), seed);
        cudaDeviceSynchronize();
    }

    Matrix forward(const Matrix& x)
    {
        input = x;
        Matrix output = Matrix::Affine(weight, x, bias);
        return output;
    }

    Matrix backward(const Matrix& delta)
    {
        Matrix test(weight.height, weight.width);
        test = weight.transpose();
        for(int row = 0;row<test.height;++row)
        {
            for(int column = 0;column<test.width;++column)
            {
                assert(test.elements[row*test.width + column] == 1);
            }
        }
        Matrix delta2(delta.width, delta.height);
        delta2 = delta;
        for(int row = 0;row<delta2.height;++row)
        {
            for(int column = 0;column<delta2.width;++column)
            {
                assert(delta2.elements[row*delta2.width + column] == 1);
            }
        }
        
        Matrix output = Matrix::Mul(weight.transpose(), delta);
        Matrix input_transpose = input.transpose();

        delta_weight  = Matrix::ScalarMul(1.0/batch_size, Matrix::Mul(delta, input_transpose));
        delta_bias    = Matrix::ScalarMul(1.0/batch_size, Matrix::Mul(delta, one_vector));

        return output;
    }  
};


struct RelU
{
    Matrix mask;
    int input_size, batch_size;
    bool onGpu;
    RelU(int _input_size, int _batch_size, bool _onGpu = false):input_size(_input_size),batch_size(_batch_size),onGpu(_onGpu),
    mask(_batch_size, _input_size, _onGpu)
    {
        
    }

    Matrix forward(const Matrix& x)
    {
        Matrix output = activate(x, mask);//relu
        return output;
    }

    Matrix backward(const Matrix& delta)
    {
        Matrix output = hide_with_mask(delta, mask);//relu
        return output;
    }

    Matrix activate(const Matrix& x, Matrix& mask)
    {
        Matrix output(x.width, x.height, x.onGpu);
        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
        dim3 dimGrid ((batch_size + dimBlock.x - 1) / dimBlock.x, (input_size + dimBlock.y - 1) / dimBlock.y);
        Kernel::RelU<<< dimGrid, dimBlock >>> (devPtr<Matrix>(x).GetDevPtr(), devPtr<Matrix>(mask).GetDevPtr(), devPtr<Matrix>(output).GetDevPtr());
        cudaDeviceSynchronize();
        return output;
    }

    Matrix hide_with_mask(const Matrix& delta, const Matrix& mask)
    {
        Matrix output(delta.width, delta.height, delta.onGpu);
        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
        dim3 dimGrid ((batch_size + dimBlock.x - 1) / dimBlock.x, (input_size + dimBlock.y - 1) / dimBlock.y);
        Kernel::hide_with_mask<<< dimGrid, dimBlock >>> (devPtr<Matrix>(delta).GetDevPtr(), devPtr<Matrix>(mask).GetDevPtr(), devPtr<Matrix>(output).GetDevPtr());
        cudaDeviceSynchronize();
        return output;
    }
};

struct SoftMaxWithLoss//lastlayer
{

    Matrix y;
    Matrix t;
    Matrix loss;
    int input_size, output_size, batch_size;
    bool onGpu;
    SoftMaxWithLoss(int _input_size, int _output_size, int _batch_size, bool _onGpu = false):input_size(_input_size), output_size(_output_size), batch_size(_batch_size),onGpu(_onGpu),
    y(_batch_size, _output_size, _onGpu),
    t(_batch_size, _output_size, _onGpu),
    loss(_batch_size, 1, _onGpu)
    {

    };

    Matrix forward(const Matrix& x, const Matrix& _t)
    {
        t = _t;
        y = softmax(x);//softmax　各列を1個の入力としてsoftmaxをかける
        loss = cross_entropy_error(y, t);

        return loss;
    }

    Matrix backward(Matrix& delta)
    {
        Matrix output = Matrix::Sub(y, t);
        return output;
    }


    //todo: measure for overflow
    //sum
    Matrix get_sum_value_par_col(const Matrix& x)
    {
        Matrix max_value(x.width, 1, x.onGpu);//max_value[col] = max(x[0][col],x[1][col], , ,)
        dim3 dimBlock(BLOCK_SIZE,x.height);
        dim3 dimGrid ((x.width + dimBlock.x - 1) / dimBlock.x);
        int two_n = 1;

        while(two_n * 2 <= x.height){
            two_n *= 2;
        }
        set_sum_value_par_col<<< dimGrid, dimBlock, dimBlock.x*dimBlock.y*sizeof(float) >>>(devPtr<Matrix>(x).GetDevPtr(), devPtr<Matrix>(max_value).GetDevPtr(), two_n);
        cudaDeviceSynchronize();
        return max_value;
    }

    //todo: measure for overflow
    Matrix softmax(const Matrix& x)
    {
        Matrix sum_value(x.width, 1, x.onGpu);//max_value[col] = max(x[0][col],x[1][col], , ,)
        Matrix max_value(x.width, 1, x.onGpu);//max_value[col] = max(x[0][col],x[1][col], , ,)
        Matrix output (x.width, x.height, x.onGpu);//max_value[col] = max(x[0][col],x[1][col], , ,)
        
        dim3 dimBlock(BLOCK_SIZE,x.height);
        dim3 dimGrid ((x.width + dimBlock.x - 1) / dimBlock.x);
        int two_n = 1;

        while(two_n * 2 <= x.height){
            two_n *= 2;
        }
        max_value = get_max_value_par_col(x);
        kernel_softmax<<< dimGrid, dimBlock, dimBlock.x*dimBlock.y*sizeof(float) >>>(devPtr<Matrix>(x).GetDevPtr(), 
        devPtr<Matrix>(sum_value).GetDevPtr(), 
        devPtr<Matrix>(max_value).GetDevPtr(),  
        devPtr<Matrix>(output).GetDevPtr(),  
        two_n);
        cudaDeviceSynchronize();
        return output;
    }

    Matrix cross_entropy_error(const Matrix& y_, const Matrix& t_)
    {
        Matrix E (y_.width, 1, onGpu);//max_value[col] = max(x[0][col],x[1][col], , ,)
        
        dim3 dimBlock(BLOCK_SIZE,y_.height);
        dim3 dimGrid ((y_.width + dimBlock.x - 1) / dimBlock.x);
        int two_n = 1;

        while(two_n * 2 <= y_.height){
            two_n *= 2;
        }
        kernel_cross_entropy<<< dimGrid, dimBlock, dimBlock.x*dimBlock.y*sizeof(float) >>>(
        devPtr<Matrix>(y_).GetDevPtr(), 
        devPtr<Matrix>(t_).GetDevPtr(), 
        devPtr<Matrix>(E).GetDevPtr(),  
        two_n);
        cudaDeviceSynchronize();
        return E;
    }

    //max
    Matrix get_max_value_par_col(const Matrix& x)
    {
        Matrix max_value(x.width, 1, x.onGpu);//max_value[col] = max(x[0][col],x[1][col], , ,)
        dim3 dimBlock(BLOCK_SIZE,x.height);
        dim3 dimGrid ((x.width + dimBlock.x - 1) / dimBlock.x);
        int two_n = 1;

        while(two_n * 2 <= x.height){
            two_n *= 2;
        }
        set_max_value_par_col<<< dimGrid, dimBlock, dimBlock.x*dimBlock.y*sizeof(float) >>>(devPtr<Matrix>(x).GetDevPtr(), devPtr<Matrix>(max_value).GetDevPtr(), two_n);
        cudaDeviceSynchronize();
        return max_value;
    }

};

int main()
{
    
}