#include <stdio.h>
#include "constants.h"

// Forward declaration of the CUDA kernel function
extern "C" __global__ void matmul_2d_tiling(float* a, float* b, float* c, const int M, const int N, const int K);

// C-compatible wrapper to call the CUDA kernel
extern "C" void matmul_2d_wrapper(float* a, float* b, float* c, const int M, const int N, const int K) {
    // Define block and grid sizes
    dim3 block(BM/TM, BN/TN, 1);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN, 1);

    // Allocate device memory
    float *device_a, *device_b, *device_c;
    cudaMalloc(&device_a, M * K * sizeof(float));
    cudaMalloc(&device_b, K * N * sizeof(float));
    cudaMalloc(&device_c, M * N * sizeof(float));

    // Transfer data to device
    cudaMemcpy(device_a, a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel with device pointers
    matmul_2d_tiling<<<grid, block>>>(device_a, device_b, device_c, M, N, K);

    // Synchronize to ensure completion
    cudaDeviceSynchronize();

    // Transfer result back to host
    cudaMemcpy(c, device_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the contents of matrix c (after kernel execution)

    // Free device memory
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
}