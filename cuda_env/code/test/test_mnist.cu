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

    void change_batch_size(int batch_size)
    {
        Affine1.input.width = batch_size;
        Affine1.one_vector.height = batch_size;
        Affine1.batch_size = batch_size;

        RelU1.batch_size = batch_size;
        RelU1.mask.width = batch_size;

        Affine2.input.width = batch_size;
        Affine2.one_vector.height = batch_size;
        Affine2.batch_size = batch_size;

        SoftMaxWithLoss1.y.width = batch_size;
        SoftMaxWithLoss1.t.width = batch_size;
        SoftMaxWithLoss1.loss.width = batch_size;
        SoftMaxWithLoss1.batch_size = batch_size;
    }

    Matrix predict_num(const Matrix& data)
    {
        return Matrix::ArgMax_par_col(predict(data));
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

    void forward_backward_test(const Matrix& x, const Matrix& t, float learning_rate)
    {
        //forward
        Matrix loss_value(2,1);
        loss_value = loss(x, t);
        assert(loss_value.elements[0] - 0.018638 < 1e-5 && loss_value.elements[1] - 1.1442640263885358e-05 < 1e-5);
        std::cout << "forward test passed\n";

        //backward
        Affine1.backward(RelU1.backward(Affine2.backward(SoftMaxWithLoss1.backward())));

        {
            Matrix test(Affine1.delta_weight.width, Affine1.delta_weight.height);
            Matrix test_ans(Affine1.delta_weight.width, Affine1.delta_weight.height);
            test = Affine1.delta_weight;
            load_data("data/grad_W1.tsv",test_ans);
            if_same_matrix(test, test_ans);
        }
        std::cout << "grad affin1_weight passed\n";

        {
            Matrix test(Affine1.delta_bias.width, Affine1.delta_bias.height);
            Matrix test_ans(Affine1.delta_bias.width, Affine1.delta_bias.height);
            test = Affine1.delta_bias;
            load_data("data/grad_b1.tsv",test_ans);
            if_same_matrix(test, test_ans);
        }
        std::cout << "grad affin1_bias passed\n";

        {
            Matrix test(Affine2.delta_weight.width, Affine2.delta_weight.height);
            Matrix test_ans(Affine2.delta_weight.width, Affine2.delta_weight.height);
            test = Affine2.delta_weight;
            load_data("data/grad_W2.tsv",test_ans);
            if_same_matrix(test, test_ans);
        }
        std::cout << "grad affin2_weight passed\n";

        {
            Matrix test(Affine2.delta_bias.width, Affine2.delta_bias.height);
            Matrix test_ans(Affine2.delta_bias.width, Affine2.delta_bias.height);
            test = Affine2.delta_bias;
            load_data("data/grad_b2.tsv",test_ans);
            if_same_matrix(test, test_ans);
        }
        std::cout << "grad affin2_bias passed\n";

        //update weight and bias
        Affine1.weight = Matrix::Add(Affine1.weight, Matrix::ScalarMul(-learning_rate, Affine1.delta_weight));
        Affine1.bias   = Matrix::Add(Affine1.bias  , Matrix::ScalarMul(-learning_rate, Affine1.delta_bias));
        Affine2.weight = Matrix::Add(Affine2.weight, Matrix::ScalarMul(-learning_rate, Affine2.delta_weight));
        Affine2.bias   = Matrix::Add(Affine2.bias  , Matrix::ScalarMul(-learning_rate, Affine2.delta_bias));


        {
            Matrix test(Affine1.delta_weight.width, Affine1.delta_weight.height);
            Matrix test_ans(Affine1.delta_weight.width, Affine1.delta_weight.height);
            test = Affine1.weight;
            load_data("data/updated_W1.tsv",test_ans);
            if_same_matrix(test, test_ans);
        }
        std::cout << "update affin1_weight passed\n";

        {
            Matrix test(Affine1.delta_bias.width, Affine1.delta_bias.height);
            Matrix test_ans(Affine1.delta_bias.width, Affine1.delta_bias.height);
            test = Affine1.bias;
            load_data("data/updated_b1.tsv",test_ans);
            if_same_matrix(test, test_ans);
        }
        std::cout << "update affin1_bias passed\n";

        {
            Matrix test(Affine2.delta_weight.width, Affine2.delta_weight.height);
            Matrix test_ans(Affine2.delta_weight.width, Affine2.delta_weight.height);
            test = Affine2.weight;
            load_data("data/updated_W2.tsv",test_ans);
            if_same_matrix(test, test_ans);
        }
        std::cout << "update affin2_weight passed\n";

        {
            Matrix test(Affine2.delta_bias.width, Affine2.delta_bias.height);
            Matrix test_ans(Affine2.delta_bias.width, Affine2.delta_bias.height);
            test = Affine2.bias;
            load_data("data/updated_b2.tsv",test_ans);
            if_same_matrix(test, test_ans);
        }
        std::cout << "update affin2_bias passed\n";
    }
};

#include <random>
std::mt19937 engine(1);

// 一様実数分布
// [-1.0, 1.0)の値の範囲で、等確率に実数を生成する
std::uniform_real_distribution<> dist1(-1.0, 1.0);
int main()
{
    Matrix h_test(20,10);
    Matrix ans(h_test.width,1);
    for(int row = 0; row < h_test.height; ++row)
        for(int col = 0; col < h_test.width; ++col)
            h_test.elements[row*h_test.width + col] = dist1(engine);

    int max_id[h_test.width];
    for(int col = 0; col < h_test.width; ++col)
    {
        float max = h_test.elements[col];
        max_id[col] = 0;
        for(int row = 1; row < h_test.height; ++row)
        {
            if(h_test.elements[row * h_test.width + col] > max) 
            {
                max = h_test.elements[row * h_test.width + col];
                max_id[col] = row;
            }
        }
    }
    Matrix d_test(h_test.width,h_test.height,true);
    d_test = h_test;
    ans = Matrix::ArgMax_par_col(d_test);
    for(int row = 0; row < ans.height; ++row)
        for(int col = 0; col < ans.width; ++col)
            assert(ans.elements[row*ans.width + col] == max_id[col]);

    
    int   input_size = 784;
    int   hidden_size = 50;
    int   output_size = 10; //classification number
    int   batch_size  = 2;
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
    load_data("data/input2.tsv", h_input);
    load_data("data/teacher2.tsv", h_teacher);

    d_input   = h_input;
    d_teacher = h_teacher;

    network.Affine1.weight = h_weight1;
    network.Affine1.bias   = h_bias1;
    network.Affine2.weight = h_weight2;
    network.Affine2.bias   = h_bias2;
    
    Matrix result(batch_size, output_size);
    result = network.predict(d_input);
    //print_matrix(result);

    network.forward_backward_test(d_input, d_teacher, learning_rate);
    assert(network.accuracy(d_input,d_teacher) == 1);
}