#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "helper_functions.h"

__global__ void kernel
        (double *vec, double scalar, unsigned int num_elements)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        vec[idx] = vec[idx] * scalar;
    }
}

void run_kernel
(double *vec, double scalar, unsigned int num_elements)
{
    size_t byte_size = num_elements * sizeof(double);

    double *gpu_ptr;
    checkCudaErrors(cudaMalloc((void**)&gpu_ptr, byte_size));
    checkCudaErrors(cudaMemcpy(gpu_ptr, vec, byte_size, cudaMemcpyHostToDevice));
    dim3 dimBlock(256, 1, 1);
    dim3 dimGrid(ceil((double)num_elements / dimBlock.x));
    kernel <<<dimGrid, dimBlock>>>
        (gpu_ptr, scalar, num_elements);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemcpy(vec, gpu_ptr, byte_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(gpu_ptr));
}