#include <assert.h>
#include <cstdio>
#include "../constants.h"

__device__ void load_GMEM(float* a, const int M, const int N, float* a_local, const int a_inner_row, const int a_inner_col) {
    for (int i=0; i < BM; i++) {
        for (int j=0; j < BN; j++) {
            a_local[i * BM + j] = a[a_inner_row * N + a_inner_col];
        }
    }
}

__device__ void load_SMEM(float* a_local, const int M, const int N, float* thread_results) {
    for (int i=0; i < TM; i++) {
        for (int j=0; j < TN; j++) {
            *thread_results += 0.0;
        }
    }
}

extern "C" __global__ void mean_squared_error(float* a, const int M, const int N) {
    const int number_of_threads = blockDim.x * blockDim.y;
    const int idx = threadIdx.x;

    __shared__ float a_local[BM * BN];
    float thread_results = 0.0;
    const int a_inner_row = idx / BM;
    const int a_inner_col = idx % BM;

    for (int block_idx=0; block_idx < M; block_idx += BN) {
        load_GMEM(a, M, N, a_local, a_inner_row, a_inner_col);
        __syncthreads();
        load_SMEM(a_local, M, N, &thread_results);
        a += BK;
        __syncthreads();
    }
}