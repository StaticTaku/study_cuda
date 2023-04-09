#include <matrix.hpp>

using namespace cuda_Matrix;
struct Affine
{
    Matrix input;
    Matrix weight;
    Matrix bias;
    Matrix delta_weight;
    Matrix delta_bias;

    int input_size, output_size, batch_size;
    bool onGpu;
    Affine(int _input_size, int _output_size, int _batch_size, bool _onGpu = false):input_size(_input_size),output_size(_output_size),batch_size(_batch_size),onGpu(_onGpu),
    weight(input_size, output_size, onGpu),
    input(batch_size, input_size, onGpu),
    bias(1, output_size, onGpu),
    delta_weight(input_size, output_size, onGpu),
    delta_bias(1, output_size, onGpu),
    {
        
    }

    Matrix forward(const Matrix& x)
    {
        input = x;
        Matrix::Affine(weight, x, b, output);
        return x;
    }

    void backward(Matrix& delta)
    {
        Matrix::Mul(delta, input_t)/batch_size;
        Matrix::Mul(delta, 1)/batch_size;
        delta = tW * delta;
    }
};

struct RelU
{
    Matrix mask;
    int input_size, output_size, batch_size;
    bool onGpu;
    RelU(int _input_size, int _output_size, int _batch_size, bool _onGpu = false):input_size(_input_size),output_size(_output_size),batch_size(_batch_size),onGpu(_onGpu),
    mask(_input_size, _output_size, _batch_size)
    {

    }

    void forward(const Matrix& x, Matrix& output)
    {
        activate(x, mask, output);//relu
    }

    void backward(Matrix& delta)
    {
        delta[mask[i]] = 0;
    }
};

struct SoftMaxWithLoss//lastlayer
{

    Matrix y;
    Matrix t;
    Matrix loss;
    __kernel__ 
    SoftMaxWithLoss(int _input_size, int _output_size, int _batch_size, bool _onGpu = false)
    {

    };

    void forward(const Matrix& x, const Matrix& _t)
    {
        t = _t;
        activate(x, y);//softmax
        loss = cross_entropy_error(y, t);
    }

    void backward(Matrix& delta)
    {
        delta = (y-t)/batch_size;    
    }
};


//Affin - RelU - Affine - SoftMaxWithLoss
//SoftMaxWithLoss.y = probability of each number
//
int main() 
{

}