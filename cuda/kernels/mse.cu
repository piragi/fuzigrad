#include <assert.h>

#include <cstdio>

#include "../constants.h"

__device__ void load_GMEM(float *a, float *b, const int M, const int N, float *a_local, float *b_local, const int inner_row, const int inner_col, const int stride) {
    for (int offset = 0; offset < MSE_BM; offset += stride) {
        reinterpret_cast<float4 *>(&a_local[(inner_row + offset) * MSE_BN + inner_col * 4])[0] = reinterpret_cast<float4 *>(&a[(inner_row + offset) * N + inner_col * 4])[0];
        reinterpret_cast<float4 *>(&b_local[(inner_row + offset) * MSE_BN + inner_col * 4])[0] = reinterpret_cast<float4 *>(&b[(inner_row + offset) * N + inner_col * 4])[0];
    }
}

__device__ void load_SMEM(float *a_local, float *b_local, const int M, const int N, float *reg_tile, const int warp_row, const int warp_col, const int warp_subtile_m, const int warp_subtile_n, const int warp_subtile_row, const int warp_subtile_col, const int inner_row, const int inner_col) {
    for (int wsm_idx = 0; wsm_idx < MSE_M_SUBTILES; wsm_idx++) {
        for (int wsn_idx = 0; wsn_idx < MSE_N_SUBTILES; wsn_idx++) {
            // TODO: cleanup
            int pos_warp = warp_row * MSE_WM * MSE_BN + warp_col * MSE_WN;
            pos_warp += (warp_subtile_row * MSE_BN * MSE_TM) + (warp_subtile_col * MSE_TN);
            int pos_subwarp = (wsm_idx * warp_subtile_m * MSE_BN) + (wsn_idx * warp_subtile_n);
            int pos_new = pos_warp + pos_subwarp;
            for (int tm_idx = 0; tm_idx < MSE_TM; tm_idx++) {
                for (int tn_idx = 0; tn_idx < MSE_TN; tn_idx += 4) {
                    // int pos = (inner_row * MSE_TM + tm_idx) * MSE_BN + inner_col * MSE_TN + tn_idx;
                    int pos = pos_new + (tm_idx * MSE_BN) + tn_idx;
                    // float a_new = a_local[pos];
                    float4 a_new = reinterpret_cast<float4 *>(&a_local[pos])[0];
                    float4 b_new = reinterpret_cast<float4 *>(&b_local[pos])[0];
                    float difference = a_new.x - b_new.x;
                    *reg_tile += difference * difference;

                    difference = a_new.y - b_new.y;
                    *reg_tile += difference * difference;

                    difference = a_new.z - b_new.z;
                    *reg_tile += difference * difference;

                    difference = a_new.w - b_new.w;
                    *reg_tile += difference * difference;
                }
            }
        }
    }
}

__device__ void shuffle_down_warps(float *reg_tile) {
    unsigned mask = __ballot_sync(0xffffffff, 1);
    for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
        *reg_tile += __shfl_down_sync(mask, *reg_tile, offset);
    }
}

extern "C" __global__ void mean_squared_error(float *a, float *b, float *block_result, const int M, const int N) {
    const int number_of_threads = blockDim.x * blockDim.y;
    const int idx = threadIdx.x;

    __shared__ float a_local[MSE_BM * MSE_BN];
    __shared__ float b_local[MSE_BM * MSE_BN];
    float reg_tile = 0.0;

    // load into SMEM through float4 into blocktile
    const int inner_row = idx / (MSE_BN / 4);
    const int inner_col = idx % (MSE_BN / 4);
    const int stride = (number_of_threads * 4) / MSE_BN;

    // thread inside warptile
    const int number_of_warps = MSE_NUMBER_OF_THREADS / WARPSIZE;
    const int warp_idx = idx / WARPSIZE;
    const int warp_row = warp_idx / (MSE_BN / MSE_WN);
    const int warp_col = warp_idx % (MSE_BN / MSE_WN);

    // thread inside warp subtile
    const int warp_subtile_idx = idx % WARPSIZE;
    const int warp_subtile_m = MSE_WM / MSE_M_SUBTILES;
    const int warp_subtile_n = MSE_WN / MSE_N_SUBTILES;
    const int warp_subtile_row = warp_subtile_idx / (warp_subtile_n / MSE_TN);
    const int warp_subtile_col = warp_subtile_idx % (warp_subtile_n / MSE_TN);

    // thread inside threadtile
    const int tile_row = idx / (MSE_BN / MSE_TN);
    const int tile_col = idx % (MSE_BN / MSE_TN);

    // bring a and b into position
    int position = blockIdx.x * MSE_BM * N + blockIdx.y * MSE_BN;
    a += position;
    b += position;
    block_result += blockIdx.x * gridDim.y + blockIdx.y;

    load_GMEM(a, b, M, N, a_local, b_local, inner_row, inner_col, stride);
    __syncthreads();
    load_SMEM(a_local, b_local, M, N, &reg_tile, warp_row, warp_col, warp_subtile_m, warp_subtile_n, warp_subtile_row, warp_subtile_col, tile_row, tile_col);
    __syncthreads();
    shuffle_down_warps(&reg_tile);
    if ((idx % 32) == 0) {
        atomicAdd(block_result, reg_tile);
    }
}