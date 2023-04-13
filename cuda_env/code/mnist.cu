#include <matrix.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <layer.hpp>
#include <fstream>

#include <fstream>
#include <string>
#include <sstream>
#include <vector>

using namespace cuda_Matrix;

struct TwoLayerNetwork
{
    Affine Affine1;
    RelU   RelU1;
    Affine Affine2;

    SoftMaxWithLoss SoftMaxWithLoss1;
    bool onGpu;
    TwoLayerNetwork(int _input_size, int _hidden_size, int _batch_size, int _output_size, float weight_init_std=0.01, bool _onGpu = true):
    Affine1(_input_size, _hidden_size, _batch_size, weight_init_std, _onGpu),
    RelU1(_hidden_size, _batch_size, _onGpu),
    Affine2(_hidden_size, _output_size, _batch_size, weight_init_std, _onGpu),
    SoftMaxWithLoss1(_output_size, _batch_size, _onGpu)
    {

    }

    Matrix predict(const Matrix& data)
    {
        Matrix result = Affine2.forward(RelU1.forward(Affine1.forward(data)));
        return result;
    }

    Matrix loss(const Matrix& data, const Matrix& teacher)
    {
        Matrix test = SoftMaxWithLoss1.forward(predict(data), teacher);
        return SoftMaxWithLoss1.forward(predict(data), teacher);
    }

    void update_weight_bias(const Matrix& x, const Matrix& t, float learning_rate)
    {
        //forward
        loss(x, t);

        //backward
        Affine1.backward(RelU1.backward(Affine2.backward(SoftMaxWithLoss1.backward())));

        //update weight and bias
        Affine1.weight = Matrix::Add(Affine1.weight, Matrix::ScalarMul(-learning_rate, Affine1.delta_weight));
        Affine1.bias   = Matrix::Add(Affine1.bias  , Matrix::ScalarMul(-learning_rate, Affine1.delta_bias));
        Affine2.weight = Matrix::Add(Affine2.weight, Matrix::ScalarMul(-learning_rate, Affine2.delta_weight));
        Affine2.bias   = Matrix::Add(Affine2.bias  , Matrix::ScalarMul(-learning_rate, Affine2.delta_bias));
    }

    float accuracy(const Matrix& data, const Matrix& teacher)
    {
        Matrix list_argmax_data(data.width, 1);
        Matrix list_argmax_teacher(teacher.width, 1);

        list_argmax_data    = Matrix::ArgMax_par_col(predict(data));
        list_argmax_teacher = Matrix::ArgMax_par_col(teacher);

        float sum = 0;
        #pragma omp parallel for reduction(+:sum)
        for(int i = 0;i<data.width;++i)
        {
            if(list_argmax_data.elements[i] - list_argmax_teacher.elements[i] < 1e-5)
                sum++;
        }

        return sum / data.width;
    }
};


int main()
{
    int   input_size = 784;
    int   hidden_size = 50;
    int   output_size = 10; //classification number
    int   batch_size  = 1;
    float learning_rate = 0.1;
    TwoLayerNetwork network(input_size, hidden_size, batch_size, output_size);

    Matrix h_weight1(input_size, hidden_size);
    Matrix h_bias1  (1, hidden_size);
    Matrix h_weight2(hidden_size, output_size);
    Matrix h_bias2  (1, output_size);
    Matrix h_input  (batch_size, input_size);
    Matrix h_teacher(batch_size, output_size);

    Matrix d_input  (batch_size, input_size, true);
    Matrix d_teacher(batch_size, output_size, true);

    load_data("data/W1.tsv", h_weight1);
    load_data("data/b1.tsv", h_bias1);
    load_data("data/W2.tsv", h_weight2);
    load_data("data/b2.tsv", h_bias2);
    load_data("data/input.tsv", h_input);
    load_data("data/teacher.tsv", h_teacher);

    d_input   = h_input;
    d_teacher = h_teacher;

    network.Affine1.weight = h_weight1;
    network.Affine1.bias   = h_bias1;
    network.Affine2.weight = h_weight2;
    network.Affine2.bias   = h_bias2;
    
    Matrix result(batch_size, output_size);
    result = network.predict(d_input);
    print_matrix(result);
    network.update_weight_bias(d_input, d_teacher, learning_rate);
}