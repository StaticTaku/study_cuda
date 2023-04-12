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
    int output_size, batch_size;
    bool onGpu;
    SoftMaxWithLoss(int _output_size, int _batch_size, bool _onGpu = false):output_size(_output_size), batch_size(_batch_size),onGpu(_onGpu),
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
    {
        int input_size = 3*16;
        int hidden_size = 50;
        int batch_size = 16;
        Affine Layer1(input_size, hidden_size, batch_size, 0.01, true);

        Matrix h_x(batch_size, input_size);
        Matrix d_x(batch_size, input_size, true);

        for(int id = 0;id<h_x.size;++id)
            h_x.elements[id] = 1;
        d_x = h_x;


        
        Matrix ans(batch_size, hidden_size);
        Matrix d_ans(batch_size, hidden_size, true);
        d_ans = Layer1.forward(d_x);
        ans = d_ans;

        for(int row = 0;row<ans.height;++row)
        {
            for(int column = 0;column<ans.width;++column)
            {
                if(ans.elements[row*ans.width + column] != input_size + row)
                {
                    std::cout << row << " " << column << "a" << ans.elements[row*ans.width + column] << std::endl;
                    std::exit(1);
                }
            }
        }

        Matrix ans_trans(hidden_size, batch_size);
        ans_trans = d_ans.transpose();
        
        for(int row = 0;row<ans_trans.height;++row)
        {
            for(int column = 0;column<ans_trans.width;++column)
            {
                if(ans_trans.elements[row*ans_trans.width + column] != input_size + column)
                {
                    std::cout << ans_trans.elements[row*ans_trans.width + column] << std::endl;
                    std::cout << row << " " << column  << " " << input_size + column << std::endl;
                    std::exit(1);
                }
            }
        }

        Matrix ddd(hidden_size, batch_size, true);
        ddd = ans_trans;
        Matrix ddd_host(hidden_size, batch_size);
        ddd_host = Matrix::ScalarMul(10, ddd);

        for(int row = 0;row<ans_trans.height;++row)
        {
            for(int column = 0;column<ans_trans.width;++column)
            {
                if(ddd_host.elements[row*ans_trans.width + column] != 10*(input_size + column))
                {
                    std::cout << ddd_host.elements[row*ans_trans.width + column] << std::endl;
                    std::cout << row << " " << column  << " " << input_size + column << std::endl;
                    std::exit(1);
                }
            }
        }

        Matrix d_delta(batch_size, hidden_size, true);
        Matrix h_delta(batch_size, hidden_size);

        for(int row = 0;row<h_delta.height;++row)
        {
            for(int column = 0;column<h_delta.width;++column)
            {
                h_delta.elements[row*h_delta.width + column] = 1;
            }
        }

        d_delta = h_delta;

        Matrix h_out(batch_size, input_size);

        h_out = Layer1.backward(d_delta);

        for(int row = 0;row<h_out.height;++row)
        {
            for(int column = 0;column<h_out.width;++column)
            {
                 assert(h_out.elements[row*h_out.width + column] == hidden_size);
            }
        }

        

    }

    {
        int input_size = 11;
        int batch_size = 14*11;
        RelU Layer2(input_size, batch_size, true);
        Matrix h_x(batch_size, input_size);
        Matrix d_x(batch_size, input_size, true);

        Matrix h_y(batch_size, input_size);
        for(int row = 0;row<h_x.height;++row)
        {
            for(int column = 0;column<h_x.width;++column)
            {
                if(column % 2 == 0)
                    h_x.elements[row*h_x.width + column] = 1;
                else
                    h_x.elements[row*h_x.width + column] = -1;
            }
        }
        d_x = h_x;
        h_y = Layer2.forward(d_x);

        for(int row = 0;row<h_x.height;++row)
        {
            for(int column = 0;column<h_x.width;++column)
            {
                if(h_x.elements[row*h_x.width + column] < 0)
                    assert(h_y.elements[row*h_x.width + column] == 0);
                else if(h_x.elements[row*h_x.width + column] > 0)
                    assert(h_y.elements[row*h_x.width + column] == 1);
            }
        }

        Matrix h_x2(batch_size, input_size);
        Matrix d_x2(batch_size, input_size, true);

        Matrix h_y2(batch_size, input_size);
        for(int row = 0;row<h_x2.height;++row)
        {
            for(int column = 0;column<h_x2.width;++column)
            {
                if(column % 2 == 0)
                    h_x2.elements[row*h_x2.width + column] = -1;
                else
                    h_x2.elements[row*h_x2.width + column] = 1;
            }
        }
        d_x2 = h_x2;

        h_y2 = Layer2.backward(d_x2);

        for(int row = 0;row<h_x.height;++row)
        {
            for(int column = 0;column<h_x.width;++column)
            {
                if(h_x.elements[row*h_x.width + column] < 0)
                    assert(h_y2.elements[row*h_x.width + column] == 0);
                else if(h_x.elements[row*h_x.width + column] > 0)
                    assert(h_y2.elements[row*h_x.width + column] == -1);
            }
        }

    }

    {
        //int input_size = 50;
        int output_size = 10;
        int batch_size = 100;
        SoftMaxWithLoss Layer3(output_size, batch_size, true);
        Matrix h_x(batch_size, output_size);
        Matrix d_x(batch_size, output_size, true);

        Matrix h_y(batch_size, 1);
        Matrix h_y_soft(batch_size, output_size);
        Matrix h_y_loss(batch_size, 1);
        for(int row = 0;row<h_x.height;++row)
        {
            for(int col = 0;col<h_x.width;++col)
            {
                h_x.elements[row * h_x.width + col] = row;
            }
        }
        d_x = h_x;
        h_y = Layer3.get_sum_value_par_col(d_x);

        for(int col = 0;col<h_y.width;++col)
        {
            assert(h_y.elements[col] == 0.5*output_size*(output_size-1));
        }

        h_y = Layer3.get_max_value_par_col(d_x);

        for(int col = 0;col<h_y.width;++col)
        {
            assert(h_y.elements[col] == (output_size-1));
        }

        h_y_soft = Layer3.softmax(d_x);

        for(int row = 0;row<h_y_soft.height;++row)
        {
            for(int col = 0;col<h_y_soft.width;++col)
            {
                //printf("%d,%d,%f\n", row, col, h_y_soft.elements[row * h_y_soft.width + col]);
            }
        }

        Matrix h_teacher(batch_size, output_size);
        Matrix d_teacher(batch_size, output_size, true);


        int ans = 9;



        for(int row = 0;row<h_teacher.height;++row)
        {
            for(int col = 0;col<h_teacher.width;++col)
            {
                h_teacher.elements[row * h_x.width + col] = row ==  ans? 1:0;
            }
        }

        d_teacher = h_teacher;
        h_y_loss = Layer3.forward(d_x, d_teacher);

        float exp_sum = 0;
        for(int i = 0;i<output_size;++i)
        {
            exp_sum += std::exp(i);
        }
        
        for(int col = 0;col<h_y_loss.width;++col)
        {
            //std::cout << -std::log(std::exp(ans)/exp_sum) << " " << h_y_loss.elements[col] << std::endl;
        }

        Matrix ans_x(batch_size, output_size);
        ans_x = Layer3.backward(d_x);

        for(int row = 0;row<h_teacher.height;++row)
        {
            for(int col = 0;col<h_teacher.width;++col)
            {
                //printf("%d,%d,%f,%f\n", row, col, ans_x.elements[row * h_x.width + col], h_y_soft.elements[row * h_x.width + col] - h_teacher.elements[row * h_x.width + col]);
                assert(ans_x.elements[row * h_x.width + col] == h_y_soft.elements[row * h_x.width + col] - h_teacher.elements[row * h_x.width + col]);
            }
        }

    }


}
