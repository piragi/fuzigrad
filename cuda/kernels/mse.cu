#include <assert.h>
#include <cstdio>
#include "../constants.h"

__device__ void load_GMEM(float* a, float* b, const int M, const int N, float* a_local, float* b_local, const int inner_row, const int inner_col, const int stride) {
    for (int offset=0; offset < BM; offset+=stride) {
        reinterpret_cast<float4*>(&a_local[(inner_row + offset) * BK + inner_col * 4])[0] 
            = reinterpret_cast<float4*>(&a[(inner_row + offset) * N + inner_col * 4])[0];

        reinterpret_cast<float4*>(&b_local[(inner_row + offset) * BK + inner_col * 4])[0] 
            = reinterpret_cast<float4*>(&b[(inner_row + offset) * N + inner_col * 4])[0];
    }
}

__device__ void load_SMEM(float* a_local, float* b_local, const int M, const int N, float* reg_tile, const int inner_row, const int inner_col) {
    for (int i=0; i < MSE_TM; i++) {
        for (int j=0; j < MSE_TN; j++) {
            float a_el = a_local[(inner_row * MSE_TM + i) * BK + inner_col * MSE_TN + j];
            float b_el = b_local[(inner_row * MSE_TM + i) * BK + inner_col * MSE_TN + j];
            float difference =  a_el - b_el;
            *reg_tile += difference * difference;
        }
    }
}

extern "C" __global__ void mean_squared_error(float* a, float* b, float* block_result, const int M, const int N) {
    const int number_of_threads = blockDim.x * blockDim.y;
    const int idx = threadIdx.x;

    __shared__ float a_local[BM * BK];
    __shared__ float b_local[BM * BK];
    float reg_tile = 0.0;
    // when every thread accesses global memory does this make things slow?
    // should i fill up registers to the max and then fill up local memory and only then fill up block_result?

    // load into SMEM
    const int inner_row = idx / (BK / 4);
    const int inner_col = idx % (BK / 4);
    const int stride = (number_of_threads * 4) / BK;

    // bring a and b into position
    int position = blockIdx.x * BM * N + blockIdx.y * BK;
    a += position;
    b += position;
    block_result += blockIdx.x * gridDim.y + blockIdx.y;

    load_GMEM(a, b, M, N, a_local, b_local, inner_row, inner_col, stride);
    __syncthreads();
    load_SMEM(a_local, b_local, M, N, &reg_tile, inner_row, inner_col);
    __syncthreads();
    atomicAdd(block_result, reg_tile);
}