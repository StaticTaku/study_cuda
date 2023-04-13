#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include <matrix.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <layer.hpp>
#include <fstream>
#include <random>
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
        Affine1.input.change_shape(batch_size, Affine1.input.height);
        Affine1.one_vector.change_shape(1, batch_size);
        Affine1.batch_size = batch_size;

        RelU1.mask.change_shape(batch_size, RelU1.mask.height);
        RelU1.batch_size = batch_size;

        Affine2.input.change_shape(batch_size, Affine2.input.height);
        Affine2.one_vector.change_shape(1, batch_size);
        Affine2.batch_size = batch_size;

        SoftMaxWithLoss1.y.change_shape(batch_size, SoftMaxWithLoss1.y.height);
        SoftMaxWithLoss1.t.change_shape(batch_size, SoftMaxWithLoss1.t.height);
        SoftMaxWithLoss1.loss.change_shape(batch_size, 1);
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
};

void set_col(const Matrix& MatA, Matrix& MatB, const std::vector<int>& set_id)
{   
    for(int col = 0;col<set_id.size();++col)
        for(int row = 0;row<MatA.height;++row)
            MatB.elements[row*MatB.width + col] = MatA.elements[row*MatA.width + set_id[col]];
}

struct RGBA {
    unsigned char r, g, b, a; //赤, 緑, 青, 透過
    RGBA() = default;
    constexpr RGBA(const unsigned char r_, const unsigned char g_, const unsigned char b_, const unsigned char a_) :r(r_), g(g_), b(b_), a(a_) {}
};

int main()
{
    int   input_size = 784;
    int   hidden_size = 50;
    int   output_size = 10; //classification number
    int   batch_size  = 100;
    int   train_size  = 60000;
    int   test_size   = 10000;
    int   iters_num   = 10000;
    float learning_rate = 0.1;
    int   iter_per_epoch = std::max(train_size / batch_size, 1);
    TwoLayerNetwork network(input_size, hidden_size, 60000, output_size);
    network.change_batch_size(batch_size);

    Matrix h_input_train  (train_size, input_size);
    Matrix h_teacher_train(train_size, output_size);
    Matrix h_input_test   (test_size , input_size);
    Matrix h_teacher_test (test_size , output_size);

    Matrix d_input_train  (train_size, input_size, true);
    Matrix d_teacher_train(train_size, output_size, true);
    Matrix d_input_test   (test_size , input_size, true);
    Matrix d_teacher_test (test_size , output_size, true);


    Matrix h_input_batch_train  (batch_size, input_size);
    Matrix h_teacher_batch_train(batch_size, output_size);

    Matrix d_input_batch_train  (batch_size, input_size, true);
    Matrix d_teacher_batch_train(batch_size, output_size, true);

    load_data("data/x_train.tsv", h_input_train);
    load_data("data/t_train.tsv", h_teacher_train);
    load_data("data/x_test.tsv", h_input_test);
    load_data("data/t_test.tsv", h_teacher_test);
    
    d_input_train   = h_input_train;
    d_teacher_train = h_teacher_train;
    d_input_test    = h_input_test;
    d_teacher_test  = h_teacher_test;

    std::random_device rnd;
    std::mt19937 mt(rnd());
    std::uniform_int_distribution<> rand_uniform_distribution(0, train_size-1);

    for(int train_num = 0;train_num < iters_num;++train_num)
    {
        std::vector<int> set_id;
        for(int i = 0;i<batch_size;++i)
            set_id.push_back(rand_uniform_distribution(mt));
        set_col(h_input_train, h_input_batch_train, set_id);
        set_col(h_teacher_train, h_teacher_batch_train, set_id);

        d_input_batch_train   = h_input_batch_train;
        d_teacher_batch_train = h_teacher_batch_train;
        network.update_weight_bias(d_input_batch_train, d_teacher_batch_train, learning_rate);

        if(train_num % iter_per_epoch == 0)
        {   
            network.change_batch_size(d_input_train.width);
            float accuracy_train = network.accuracy(d_input_train, d_teacher_train);
            network.change_batch_size(d_input_test.width);
            float accuracy_test  = network.accuracy(d_input_test , d_teacher_test );
            network.change_batch_size(batch_size);
            printf("At %d epoc: train_accuracy = %f, test_accuracy = %f\n", train_num / iter_per_epoch, accuracy_train, accuracy_test);
        }
    }

    {
        int pic_id;
        Matrix pic(28,28);
        Matrix h_input_data(1,input_size);
        Matrix d_input_data(1,input_size, true);
        constexpr std::size_t width{ 28 }, height{ 28 }; //幅と高さ
        network.change_batch_size(1);
        while (true)
        {
            std::cout << "input a picture number from 0 to 9999 or -100 to stop this program\n";
            std::cin  >>  pic_id;
            if(!(0 <= pic_id && pic_id < test_size) && pic_id != -100)
            {
                std::cout << "invalid picture number\n";
                continue;
            }else if(pic_id == -100)
            {
                return 0;
            }
            
            for(int row = 0;row < pic.height; ++row)
                for(int col = 0;col < pic.width; ++col)
                    pic.elements[row*pic.width + col] = h_input_test.elements[(row*pic.width + col)*h_input_test.width + pic_id];

            for(int row = 0;row < h_input_data.height; ++row)
                for(int col = 0;col < h_input_data.width; ++col)
                    h_input_data.elements[row*h_input_data.width + col] = h_input_test.elements[row*h_input_test.width + pic_id];
            d_input_data = h_input_data;

            std::unique_ptr<RGBA[][width]> rgba(new(std::nothrow) RGBA[height][width]);
            if (!rgba) return -1;

            for (std::size_t row{}; row < height; ++row)
                for (std::size_t col{}; col < width; ++col) 
                {
                    rgba[row][col].r = pic.elements[row*pic.width + col]*255;
                    rgba[row][col].g = pic.elements[row*pic.width + col]*255;
                    rgba[row][col].b = pic.elements[row*pic.width + col]*255;
                    rgba[row][col].a = 255; //不透過
                }

            stbi_write_png("picture_display.png", static_cast<int>(width), static_cast<int>(height), static_cast<int>(sizeof(RGBA)), rgba.get(), 0);
            std::cout << "displayed in picture_display.png!\n";

            Matrix ans(1,output_size);
            ans = network.predict(d_input_data);

            int num = 0;
            float max = -1e5;
            for(int i = 0; i<output_size; ++i)
            {
                if(ans.elements[i] > max)
                {
                    max = ans.elements[i];
                    num = i;
                }
            }

            printf("I think the number is %d!\n\n\n", num);
        }
    }
}