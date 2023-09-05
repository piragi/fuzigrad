#include <cstdio>
#include <cuda_runtime.h>
#include <assert.h>
#include "constants.h"


#define CUDA_CHECK_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


// Declaration of the custom CUDA kernel function (adjust as needed)
extern "C" __global__ void matmul_2d_tiling(float* a, float* b, float* c, const int M, const int N, const int K);

// Custom matrix multiplication function
extern "C" void matmul(float* a, float* b, float* c, const int M, const int N, const int K) {
    float* d_a, * d_b, * d_c;

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
    matmul_2d_tiling << <grid, block >> > (d_a, d_b, d_c, M, N, K);
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
    float* d_a, * d_b, * d_c;

    cudaMalloc((void**)&d_a, sizeof(float) * M * N);
    cudaMalloc((void**)&d_b, sizeof(float) * M * N);
    cudaMalloc((void**)&d_c, sizeof(float) * 8);

    cudaMemcpy(d_a, a, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    dim3 block(NUMBER_OF_THREADS);
    dim3 grid((M + BM - 1) / BM, (N + BK - 1) / BK);
    mean_squared_error << <grid, block >> > (d_a, d_b, d_c, M, N);
    CUDA_CHECK_ERROR(cudaPeekAtLastError());
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, sizeof(float) * 8, cudaMemcpyDeviceToHost);

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}