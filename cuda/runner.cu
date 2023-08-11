#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <assert.h>
#include <cmath>
#include "matmul_2d.cu"

void matmul_cuda(float *h_a, float *h_b, float *h_c, int M, int N, int K) {
    float *d_a, *d_b, *d_c;

    cudaMalloc((void **)&d_a, sizeof(float) * M * K);
    cudaMalloc((void **)&d_b, sizeof(float) * K * N);
    cudaMalloc((void **)&d_c, sizeof(float) * M * N);

    cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    dim3 block(BM / TM, BN / TN);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    matmul_2d_tiling<<<grid, block>>>(d_a, d_b, d_c, M, N, K);

    cudaMemcpy(h_c, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void sgemm_cublas(float *h_a, float *h_b, float *h_c_blas, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, sizeof(float) * M * K);
    cudaMalloc((void **)&d_b, sizeof(float) * K * N);
    cudaMalloc((void **)&d_c, sizeof(float) * M * N);

    cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);

    cudaMemcpy(h_c_blas, d_c, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cublasDestroy(handle);
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const int num_iterations = 100; // Number of iterations for timing

    float *h_a = (float *)malloc(sizeof(float) * M * K);
    float *h_b = (float *)malloc(sizeof(float) * K * N);
    float *h_c = (float *)malloc(sizeof(float) * M * N);
    float *h_c_blas = (float *)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M * K; i++) h_a[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_b[i] = (float)rand() / RAND_MAX;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Timing custom kernel
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        matmul_cuda(h_a, h_b, h_c, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float gflops_custom = (2.0f * M * N * K * num_iterations) / (milliseconds * 1e6);
    printf("Custom kernel: %f ms, %f GFLOPS\n", milliseconds, gflops_custom);

    // Timing cuBLAS
    cudaEventRecord(start);
    for (int i = 0; i < num_iterations; i++) {
        sgemm_cublas(h_a, h_b, h_c_blas, M, N, K);
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
    printf("Max Error: %f\n", maxError);

    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_blas);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}