#include <matrix.hpp>
#include <curand.h>
#include <curand_kernel.h>

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
    int id  = blockIdx.x * blockDim.x + threadIdx.x;;
    if (id < _bias->size)
        _bias->elements[id] = id;
}

__global__ 
void set_one_vector(Matrix* _bias, int seed)
{
    int id  = blockIdx.x * blockDim.x + threadIdx.x;;
    if (id < _bias->size)
        _bias->elements[id] = 1;
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
    one_vector(1, _output_size, _onGpu)
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
    SoftMaxWithLoss(int _input_size, int _output_size, int _batch_size, bool _onGpu = false):
    y(_batch_size, _output_size, _onGpu),
    t(_batch_size, _output_size, _onGpu),
    loss(_batch_size, 1, _onGpu)
    {

    };

    /*Matrix forward(const Matrix& x, const Matrix& _t)
    {
        t = _t;
        Matrix output = activate(x, y);//softmax
        loss = cross_entropy_error(y, t);

        return loss;
    }

    Matrix backward(Matrix& delta)
    {
        Matrix output = Matrix::Sub(y, t);
        return output;
    }*/
};


//Affin - RelU - Affine - SoftMaxWithLoss
//SoftMaxWithLoss.y = probability of each number
//
/*int main() 
{
    int input_size = 12*20;
    int hidden_size = 50;
    int batch_size = 2*16;
    bool onGPU = true;
    Affine Layer1(input_size, hidden_size, batch_size, 0.01, onGPU);
    Matrix h_x(batch_size, input_size);
    Matrix d_x(batch_size, input_size, onGPU);

    for(int id = 0;id<h_x.size;++id)
        h_x.elements[id] = 1;
    
    Matrix test(batch_size, hidden_size);
    d_x = h_x;
    //test = Layer1.forward(d_x);
    test = Matrix::Mul( Layer1.weight, d_x);
    Matrix tt(input_size, hidden_size);
    tt = Layer1.weight;

    
    for(int row = 0;row<test.height;++row)
    {
        for(int column = 0;column<test.width;++column)
        {
            if(test.elements[row*test.width + column] != 12*20 )
            {
                std::cout << row << " " << column << " " << test.elements[row*test.width + column] << std::endl;
                std::exit(1);
            }
        }
    }
    return 0;

    for(int id = 0;id<tt.size;++id)
        if(tt.elements[id] != 1)
            std::cout << tt.elements[id] << std::endl;

    for(int id = 0;id<h_x.size;++id)
        if(h_x.elements[id] != 1)
            std::cout << h_x.elements[id] << std::endl;
}*/

int main()
{
    {
        int input_size = 3*16;
        int hidden_size = 50;
        int batch_size = 14*16;
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


}
