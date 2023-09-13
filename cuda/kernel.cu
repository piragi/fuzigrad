#include <assert.h>
#include <cuda_runtime.h>

#include <cstdio>

#include "constants.h"

#define CUDA_CHECK_ERROR(call)                                                                         \
    do {                                                                                               \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess) {                                                                      \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    } while (0)

// Declaration of the custom CUDA kernel function (adjust as needed)
extern "C" __global__ void matmul_2d_tiling(float* a, float* b, float* c, const int M, const int N, const int K);

// Custom matrix multiplication function
extern "C" void matmul(float* a, float* b, float* c, const int M, const int N, const int K) {
    float *d_a, *d_b, *d_c;

    assert(BM == BN);
    assert(((BK * BM) / NUMBER_OF_THREADS) % 4 == 0);
    assert((WM * WN) / (TM * TN * WARPSIZE) >= N_SUBTILES);

    cudaMalloc((void**)&d_a, sizeof(float) * M * K);
    cudaMalloc((void**)&d_b, sizeof(float) * K * N);
    cudaMalloc((void**)&d_c, sizeof(float) * M * N);

    cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    dim3 block(NUMBER_OF_THREADS);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    matmul_2d_tiling<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// Declaration of the custom CUDA kernel function (adjust as needed)
extern "C" __global__ void mean_squared_error(float* a, float* b, float* c, const int M, const int N);

// Custom matrix multiplication function
extern "C" void mse(float* a, float* b, float* c, const int M, const int N) {
    float *d_a, *d_b, *d_c;

    dim3 block(MSE_NUMBER_OF_THREADS);
    dim3 grid((M + MSE_BM - 1) / MSE_BM, (N + MSE_BN - 1) / MSE_BN);
    printf("M: %d, N: %d -- %d blocks and %d threads per block\n", M, N, grid.x * grid.y, block.x);

    cudaMalloc((void**)&d_a, sizeof(float) * M * N);
    cudaMalloc((void**)&d_b, sizeof(float) * M * N);
    cudaMalloc((void**)&d_c, sizeof(float) * grid.x * grid.y);

    cudaMemcpy(d_a, a, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    mean_squared_error<<<grid, block>>>(d_a, d_b, d_c, M, N);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, sizeof(float) * grid.x * grid.y, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

// Declaration of the custom CUDA kernel function (adjust as needed)
extern "C" __global__ void reduce_warps(float* a, const int M, float* result);

// Custom matrix multiplication function
extern "C" void reduce_kernel(float* a, const int M, float* result) {
    float* d_a;
    float* d_result;

    dim3 block(REDUCE_NUMBER_OF_THREADS);
    dim3 grid((M + REDUCE_BM - 1) / REDUCE_BM);
    printf("M: %d -- %d block(s) and %d threads per block\n", M, grid.x * grid.y, block.x);

    cudaMalloc((void**)&d_a, sizeof(float) * M);
    cudaMalloc((void**)&d_result, sizeof(float) * grid.x);

    cudaMemcpy(d_a, a, sizeof(float) * M, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, result, sizeof(float) * grid.x, cudaMemcpyHostToDevice);

    reduce_warps<<<grid, block>>>(d_a, M, d_result);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    cudaMemcpy(result, d_result, sizeof(float) * grid.x, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_result);
}