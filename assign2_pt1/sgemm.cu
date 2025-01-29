#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cublas_v2.h>


// Keep in mind, A is (m,d), B is (d,n), C is (m,n)
// Though we are dealing square matrices

__global__ void matmul_v1(float* A, float* B, float* C, const int m, const int n, const int d){
    /* TODO IMPLEMENT THIS KERNEL */

}

__global__ void matmul_v2(float* A, float* B, float* C, const int m, const int n, const int d){
    /* TODO IMPLEMENT THIS KERNEL */

}

__global__ void matmul_v3(float* A, float* B, float* C, const int m, const int n, const int d){
    /* TODO IMPLEMENT THIS KERNEL */

}

__global__ void matmul_v4(float* A, float* B, float* C, const int m, const int n, const int d){
    /* TODO IMPLEMENT THIS KERNEL */

}

torch::Tensor launch_matmul_v1(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int m, const int n, const int d){
    const int block_size = //TODO;
    dim3 blockDim(// TODO);

    const int grid_size_x = //TODO; // hint: spawn enough blocks to process m dimension
    const int grid_size_y = // TODO; // hint: spawn enough blocks to process n dimension
    dim3 gridDim(//TODO);

    matmul_v1<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        m, n, d
    );
    return C;
}

torch::Tensor launch_matmul_v2(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int m, const int n, const int d){
    /* TODO IMPLEMENT THIS KERNEL WRAPPER */
}

torch::Tensor launch_matmul_v3(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int m, const int n, const int d){
    /* TODO IMPLEMENT THIS KERNEL WRAPPER */

}

torch::Tensor launch_matmul_v4(torch::Tensor A, torch::Tensor B, torch::Tensor C, const int m, const int n, const int d){
    /* TODO IMPLEMENT THIS KERNEL WRAPPER */

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_v1", &launch_matmul_v1, "Naive Matmul");
  m.def("matmul_v2", &launch_matmul_v2, "Coalesced Matmul");
  m.def("matmul_v3", &launch_matmul_v3, "Register Reuse Matmul");
  m.def("matmul_v4", &launch_matmul_v4, "Shared Mem Block Matmul");
}

