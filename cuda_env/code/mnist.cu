#include <matrix.hpp>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <layer.hpp>

using namespace cuda_Matrix;

struct TwoLayerNetwork
{
    Affine Affine1;
    RelU   RelU1;
    Affine Affine2;

    SoftMaxWithLoss SoftMaxWithLoss1;
    bool onGpu;
    TwoLayerNetwork(int _input_size, int _hidden_size, int _batch_size, int _output_size, bool _onGpu = true, float weight_init_std=0.01):
    Affine1(_input_size, _hidden_size, _batch_size, _onGpu),
    RelU1(_hidden_size, _batch_size, _onGpu),
    Affine2(_hidden_size, _output_size, _batch_size, _onGpu),
    SoftMaxWithLoss1(_output_size, _batch_size, _onGpu)
    {

    }

};

int main()
{
    int input_size = 784;
    int hidden_size = 50;
    int output_size = 10;
    int batch_size  = 100;
    TwoLayerNetwork network();
}