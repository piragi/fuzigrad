#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "constants.h"
#include "kernel.h"

// cuBLAS matrix multiplication function
void matmul_cublas(float* a, float* b, float* c, const int M, const int N, const int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float* d_a, * d_b, * d_c;
    cudaMalloc((void**)&d_a, sizeof(float) * M * K);
    cudaMalloc((void**)&d_b, sizeof(float) * K * N);
    cudaMalloc((void**)&d_c, sizeof(float) * M * N);

    cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);

    cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cublasDestroy(handle);
}

// Benchmarking function
extern "C" void matmul_benchmark(float* a, float* b, float* c, const int M, const int N, const int K) {
    const int num_iterations = 50; // Number of iterations for timing

    float* h_a = (float*)malloc(sizeof(float) * M * K);
    float* h_b = (float*)malloc(sizeof(float) * K * N);
    float* h_c = (float*)malloc(sizeof(float) * M * N);
    float* h_c_blas = (float*)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M * K; i++) h_a[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_b[i] = (float)rand() / RAND_MAX;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timing custom kernel
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        matmul(h_a, h_b, h_c, M, N, K);
    }
    cudaEventRecord(stop);

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float tflops_custom = (2.0f * M * N * K * num_iterations * 1e-9) / (milliseconds);
    printf("Custom kernel: %f ms, %f TFLOPS\n", milliseconds, tflops_custom);

    // Timing cuBLAS
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        matmul_cublas(h_a, h_b, h_c_blas, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float tflops_cublas = (2.0f * M * N * K * num_iterations * 1e-9) / (milliseconds);
    printf("cuBLAS: %f ms, %f TFLOPS\n", milliseconds, tflops_cublas);

    float maxError = 0.0f;
    for (int i = 0; i < M * N; i++) {
        maxError = fmax(maxError, fabs(h_c[i] - h_c_blas[i]));
    }

    printf("Max Error: %f\n", maxError);
    printf("Perf. Difference to cuBLAS: %f%%\n", (tflops_custom / tflops_cublas) * 100.0f);
    printf("----\n");
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_blas);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Wrapper function to be called from Python
extern "C" void matmul_benchmark_wrapper(float* a, float* b, float* c, const int M, const int N, const int K) {
    matmul_benchmark(a, b, c, M, N, K);
}
