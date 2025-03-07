#include <cuda.h>
#include <torch/extension.h>

template <typename scalar_t>
__global__ void naive_softmax(const scalar_t* x, scalar_t* y, const int bz, const int num_classes){
  //input: (b,c)
  int i = blockIdx.y*blockDim.y + threadIdx.y;
  int j = blockIdx.x*blockDim.x + threadIdx.x;

  if(i < bz && j < num_classes){
    float running_max = -INFINITY;
    for(int c = 0; c < num_classes; c++){
      running_max = fmaxf(running_max, x[i*num_classes+c]);
    }
    float running_sum = 0;
    for(int c = 0; c < num_classes; c++){
      running_sum += __expf(x[i*num_classes+c] - running_max);
    }
    y[i*num_classes+j] = (__expf(x[i*num_classes+j] - running_max))/running_sum;
  }
}

torch::Tensor launch_naive_softmax(torch::Tensor x, torch::Tensor y, const int bz, const int num_classes){
  const int BLOCK_SIZE = 16;
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((num_classes + BLOCK_SIZE - 1)/BLOCK_SIZE, (bz + BLOCK_SIZE - 1)/BLOCK_SIZE);
  naive_softmax<<<gridDim, blockDim>>>(
    x.data_ptr<float>(),
    y.data_ptr<float>(),
    bz, num_classes 
  );
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  m.def("naive_softmax_forward", &launch_naive_softmax, "Naive Softmax");
}
