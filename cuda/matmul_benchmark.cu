#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <assert.h>
#include "constants.h"

// Declaration of the custom CUDA kernel function (adjust as needed)
extern "C" __global__ void matmul_2d_tiling(float* a, float* b, float* c, const int M, const int N, const int K, int* flag);

// Custom matrix multiplication function
void matmul_custom(float* a, float* b, float* c, const int M, const int N, const int K) {
    float* d_a, * d_b, * d_c;
    const int NUMBER_OF_THREADS = 256;


    assert(BM == BN);
    assert((BK * BM) / NUMBER_OF_THREADS % 4 == 0);

    cudaMalloc((void**)&d_a, sizeof(float) * M * K);
    cudaMalloc((void**)&d_b, sizeof(float) * K * N);
    cudaMalloc((void**)&d_c, sizeof(float) * M * N);

    cudaMemcpy(d_a, a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    int h_flag = 0;
    int* d_flag;
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemcpy(d_flag, &h_flag, sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(BM / TM, BN / TN);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    matmul_2d_tiling << <grid, block >> > (d_a, d_b, d_c, M, N, K, d_flag);

    cudaMemcpy(c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_flag) {
        printf("Zero value encountered in thread_results\n");
    }

    // Clean up
    cudaFree(d_flag);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

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
    const int num_iterations = 100; // Number of iterations for timing

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
        matmul_custom(h_a, h_b, h_c, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float gflops_custom = (2.0f * M * N * K * num_iterations) / (milliseconds * 1e6);
    printf("Custom kernel: %f ms, %f GFLOPS\n", milliseconds, gflops_custom);

    int count = 0;
    int size = 0;
    for (int i = 0; i < M * N; i++) {
        size++;
        if (h_c[i] == 1.0f) {
            count++;
        }
    }

    printf("Number of ones: %d\n", count);
    printf("Number of elements: %d\n", size);
    printf("Number of thread_result total: %f\n", h_c[0]);

    // Timing cuBLAS
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        matmul_cublas(h_a, h_b, h_c_blas, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    float gflops_cublas = (2.0f * M * N * K * num_iterations) / (milliseconds * 1e6);
    printf("cuBLAS: %f ms, %f GFLOPS\n", milliseconds, gflops_cublas);

    float maxError = 0.0f;
    for (int i = 0; i < M * N; i++) {
        maxError = fmax(maxError, fabs(h_c[i] - h_c_blas[i]));
    }
    bool first_element_correct = false;
    if (fabs(h_c[0] - h_c_blas[0]) < 1e-5f) {
        first_element_correct = true;
    }

    // weird that thats the error?!
    int count_correct_elements = 0;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_c[i] - h_c_blas[i]) <= 1e-5f) {
            count_correct_elements++;
        }
    }

    printf("First element is correct: %d\n", first_element_correct);
    printf("Number of correct elements: %d\n", count_correct_elements);
    printf("Max Error: %f\n", maxError);
    printf("Perf. Difference to cuBLAS: %f%%\n", (gflops_custom / gflops_cublas) * 100.0f);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.f ", fabs(h_c[i] - h_c_blas[i]));
        }
        printf("\n");
    }

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