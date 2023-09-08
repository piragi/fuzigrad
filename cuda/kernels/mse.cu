#include <assert.h>

#include <cstdio>

#include "../constants.h"

__device__ void load_GMEM(float *a, float *b, const int M, const int N, float *a_local, float *b_local, const int inner_row, const int inner_col, const int stride) {
    for (int offset = 0; offset < MSE_BM; offset += stride) {
        reinterpret_cast<float4 *>(&a_local[(inner_row + offset) * MSE_BN + inner_col * 4])[0] = reinterpret_cast<float4 *>(&a[(inner_row + offset) * N + inner_col * 4])[0];
        reinterpret_cast<float4 *>(&b_local[(inner_row + offset) * MSE_BN + inner_col * 4])[0] = reinterpret_cast<float4 *>(&b[(inner_row + offset) * N + inner_col * 4])[0];
    }
}

__device__ void load_SMEM(float *a_local, float *b_local, const int M, const int N, float *reg_tile, const int inner_row, const int inner_col) {
    for (int i = 0; i < MSE_TM; i++) {
        for (int j = 0; j < MSE_TN; j++) {
            int pos = (inner_row * MSE_TM + i) * MSE_BN + inner_col * MSE_TN + j;
            float difference = a_local[pos] - b_local[pos];
            *reg_tile += difference * difference;
        }
    }
}

extern "C" __global__ void mean_squared_error(float *a, float *b, float *block_result, const int M, const int N) {
    const int number_of_threads = blockDim.x * blockDim.y;
    const int idx = threadIdx.x;

    __shared__ float a_local[MSE_BM * MSE_BN];
    __shared__ float b_local[MSE_BM * MSE_BN];
    float reg_tile = 0.0;

    // load into SMEM through float4
    const int inner_row = idx / (MSE_BN / 4);
    const int inner_col = idx % (MSE_BN / 4);
    const int stride = (number_of_threads * 4) / MSE_BN;

    // thread inside a blocktile
    const int tile_row = idx / (MSE_BN / MSE_TN);
    const int tile_col = idx % (MSE_BN / MSE_TN);

    // bring a and b into position
    int position = blockIdx.x * MSE_BM * N + blockIdx.y * MSE_BN;
    a += position;
    b += position;
    block_result += blockIdx.x * gridDim.y + blockIdx.y;

    load_GMEM(a, b, M, N, a_local, b_local, inner_row, inner_col, stride);
    __syncthreads();
    load_SMEM(a_local, b_local, M, N, &reg_tile, tile_row, tile_col);
    __syncthreads();
    atomicAdd(block_result, reg_tile);
}