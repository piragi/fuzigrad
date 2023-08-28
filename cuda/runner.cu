#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <assert.h>
#include <cmath>
#include "constants.h"
#include "matmul_2d.h"
#include <iostream>
#include <cmath>
#include <map>

void matmul_cuda(float* h_a, float* h_b, float* h_c, int M, int N, int K) {
    float* d_a, * d_b, * d_c;
    // (BM * BN) / (WM * WN) * WARPSIZE = Number of threads per block
    int number_of_threads = 128;

    cudaMalloc((void**)&d_a, sizeof(float) * M * K);
    cudaMalloc((void**)&d_b, sizeof(float) * K * N);
    cudaMalloc((void**)&d_c, sizeof(float) * M * N);

    cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    dim3 block(number_of_threads);
    dim3 grid(((M + (BM - 1)) / BM), ((N + (BN - 1)) / BN));

    matmul_2d_tiling << <grid, block >> > (d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void matmul_cublas(float* h_a, float* h_b, float* h_c_cublas, int M, int N, int K) {
    float* d_a, * d_b, * d_c;

    cudaMalloc((void**)&d_a, sizeof(float) * M * K);
    cudaMalloc((void**)&d_b, sizeof(float) * K * N);
    cudaMalloc((void**)&d_c, sizeof(float) * M * N);

    cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);

    cudaMemcpy(h_c_cublas, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void compare_arrays(float* a, float* b, int size) {
    std::map<float, int> error_counts; // To store unique errors and their counts

    for (int i = 0; i < size; i++) {
        double error = std::fabs(a[i] - b[i]);
        if (error > 0.0001) { // You may adjust the tolerance level as needed
            error_counts[error]++;
        }
    }

    // Print the unique errors and their counts
    printf("Unique Errors and Their Counts:\n");
    for (const auto& [error, count] : error_counts) {
        printf("Error: %f, Count: %d\n", error, count);
    }
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;

    float* h_a = (float*)malloc(sizeof(float) * M * K);
    float* h_b = (float*)malloc(sizeof(float) * K * N);
    float* h_c = (float*)malloc(sizeof(float) * M * N);
    float* h_c_cublas = (float*)malloc(sizeof(float) * M * N);

    srand(10);
    for (int i = 0; i < M * K; i++) h_a[i] = (float)(rand() % 9);
    for (int i = 0; i < K * N; i++) h_b[i] = (float)(rand() % 9);

    matmul_cuda(h_a, h_b, h_c, M, N, K);
    matmul_cublas(h_a, h_b, h_c_cublas, M, N, K);

    // print total error
    float total_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        total_error += std::abs(h_c[i] - h_c_cublas[i]);
    }
    printf("Total Error: %f\n", total_error);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_cublas);

    return 0;
}

