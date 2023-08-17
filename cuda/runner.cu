#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <assert.h>
#include <cmath>
#include "constants.h"
#include "matmul_2d.h"

void matmul_cuda(float* h_a, float* h_b, float* h_c, int M, int N, int K) {
    float* d_a, * d_b, * d_c;
    int number_of_threads = 128;

    cudaMalloc((void**)&d_a, sizeof(float) * M * K);
    cudaMalloc((void**)&d_b, sizeof(float) * K * N);
    cudaMalloc((void**)&d_c, sizeof(float) * M * N);

    cudaMemcpy(d_a, h_a, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sizeof(float) * K * N, cudaMemcpyHostToDevice);

    dim3 block(BM / TM, BN / TN);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);

    matmul_2d_tiling << <grid, block >> > (d_a, d_b, d_c, M, N, K, 0, 0, 0, 0);

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

int main() {
    const int M = 128;
    const int N = 128;
    const int K = 128;
    const int num_iterations = 1; // Number of iterations for timing

    float* h_a = (float*)malloc(sizeof(float) * M * K);
    float* h_b = (float*)malloc(sizeof(float) * K * N);
    float* h_c = (float*)malloc(sizeof(float) * M * N);
    float* h_c_cublas = (float*)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M * K; i++) h_a[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) h_b[i] = (float)rand() / RAND_MAX;

    matmul_cuda(h_a, h_b, h_c, M, N, K);
    matmul_cublas(h_a, h_b, h_c_cublas, M, N, K);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.f ", fabs(h_c[i] - h_c_cublas[i]));
        }
        printf("\n");
    }

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}