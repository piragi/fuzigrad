#include <assert.h>
#include <cstdio>
#include "constants.h"

__device__ void load_GMEM(float* a_local, float* b_local, float* a, float* b, const int N, const int K, int a_stride, int a_inner_row, int a_inner_col, int b_stride, int b_inner_row,
    int b_inner_col) {
    // load 4 floats into local memory (L1 SMEM) 
    // number of elements per tile / number of threads need to be dividable by 4
    // (BK * BM) / (number_of_threads) % 4 == 0 
    // meaning 1024 / 256 = 4
    // to achieve a BK != TM we need to load multiple float4s
    // meaning one thread loads float4 from row x and row x + offset (for 2048 elements with 256 threads)
    for (int offset = 0; offset < BM; offset += a_stride) {
        float4 tmp = reinterpret_cast<float4*>(&a[(a_inner_row + offset) * K + a_inner_col * 4])[0];
        a_local[(a_inner_col * 4 + 0) * BM + a_inner_row + offset] = tmp.x;
        a_local[(a_inner_col * 4 + 1) * BM + a_inner_row + offset] = tmp.y;
        a_local[(a_inner_col * 4 + 2) * BM + a_inner_row + offset] = tmp.z;
        a_local[(a_inner_col * 4 + 3) * BM + a_inner_row + offset] = tmp.w;
    }
    for (int offset = 0; offset < BK; offset += b_stride) {
        reinterpret_cast<float4*>(&b_local[(b_inner_row + offset) * BN + b_inner_col * 4])[0] =
            reinterpret_cast<float4*>(&b[(b_inner_row + offset) * N + b_inner_col * 4])[0];
    }

}

__device__ void load_SMEM(float* a_local, float* b_local, float* regM, float* regN, float* thread_results, const int thread_row_subtile, const int thread_col_subtile,
    const int wm_subtile, const int wn_subtile, const int m_subtiles, const int n_subtiles, const int warp_row, const int warp_col) {
    // each thread goes through one TM*TN tiles inside the SMEM block
    // load into registers and compute thread_results
    // each thread computes TM*TN element per warp subtile

    // what warp am I in?
    // what subtile am I in?
    // position in warp = warprow * WM + warpcol
    // position in subtile = wn
    for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
        for (int wm_idx = 0; wm_idx < m_subtiles; wm_idx++) {
            for (int a_idx = 0; a_idx < TM; a_idx++) {
                int pos_warp = warp_row * WM;
                int pos_warp_subtile = wm_idx * wm_subtile;
                regM[wm_idx * TM + a_idx] = a_local[(dot_idx * BM) + pos_warp + pos_warp_subtile + thread_row_subtile * TM + a_idx];
            }
        }
        for (int wn_idx = 0; wn_idx < n_subtiles; wn_idx++) {
            for (int b_idx = 0; b_idx < TN; b_idx++) {
                int pos_warp = warp_col * WN;
                int pos_warp_subtile = wn_idx * wn_subtile;
                regN[wn_idx * TN + b_idx] = b_local[(dot_idx * BN) + pos_warp + pos_warp_subtile + thread_col_subtile * TN + b_idx];
            }
        }

        for (int wm_idx = 0; wm_idx < m_subtiles; wm_idx++) {
            for (int wn_idx = 0; wn_idx < n_subtiles; wn_idx++) {
                for (int res_idx_m = 0; res_idx_m < TM; res_idx_m++) {
                    for (int res_idx_n = 0; res_idx_n < TN; res_idx_n++) {
                        thread_results[(wm_idx * TM + res_idx_m) * (TN * n_subtiles) + (wn_idx * TN) + res_idx_n] +=
                            regM[wm_idx * TM + res_idx_m] * regN[wn_idx * TN + res_idx_n];
                    }
                }
            }
        }
    }
}


extern "C" __global__ void matmul_2d_tiling(float* a, float* b, float* c, const int M, const int N, const int K) {
    const int number_of_threads = blockDim.x * blockDim.y;

    // position of block in grid
    const int c_row = blockIdx.y;
    const int c_col = blockIdx.x;

    // position of thread in block
    const int idx = threadIdx.x;

    // position of warp in block
    const int warp_idx = idx / WARPSIZE;
    const int warp_row = warp_idx / (BN / WN);
    const int warp_col = warp_idx % (BN / WN);

    __shared__ float a_local[BM * BK];
    __shared__ float b_local[BK * BN];

    const int a_inner_col = idx % (BK / 4);
    const int a_inner_row = idx / (BK / 4);
    const int a_stride = (number_of_threads * 4) / BK;

    const int b_inner_col = idx % (BN / 4);
    const int b_inner_row = idx / (BN / 4);
    const int b_stride = (number_of_threads * 4) / BN;

    const int number_of_warps = number_of_threads / WARPSIZE;

    const int n_subtiles = 2;
    const int m_subtiles = (WM * WN) / (WARPSIZE * TM * TN * n_subtiles);
    const int wn_subtile = WN / n_subtiles;
    const int wm_subtile = WM / m_subtiles;

    const int thread_idx_subtile = idx % (WARPSIZE);
    const int thread_row_subtile = thread_idx_subtile / (wn_subtile / TN);
    const int thread_col_subtile = thread_idx_subtile % (wn_subtile / TN);

    float thread_results[TM * TN * n_subtiles * m_subtiles] = { 0.0 };
    float regM[TM * m_subtiles] = { 0.0 };
    float regN[TN * n_subtiles] = { 0.0 };

    // move a and b to correct position
    a += c_row * BM * K;
    b += c_col * BN;
    // move c to warptile
    c += (c_row * BM + warp_row * WM) * N + c_col * BN + warp_col * WN;

    for (int block_idx = 0; block_idx < K; block_idx += BK) {
        load_GMEM(a_local, b_local, a, b, N, K, a_stride, a_inner_row, a_inner_col, b_stride, b_inner_row, b_inner_col);
        __syncthreads();
        // move a tile sideways
        // move b tile downwards
        load_SMEM(a_local, b_local, regM, regN, thread_results, thread_row_subtile, thread_col_subtile, wm_subtile, wn_subtile, m_subtiles, n_subtiles, warp_row, warp_col);
        a += BK;
        b += BK * N;
        __syncthreads();
    }

    // write into GMEM
    for (int wm_idx = 0; wm_idx < m_subtiles; wm_idx++) {
        for (int wn_idx = 0; wn_idx < n_subtiles; wn_idx++) {
            float* c_interim = c + (wm_idx * wm_subtile) * N + wn_idx * wn_subtile;
            for (int res_idx_m = 0; res_idx_m < TM; res_idx_m++) {
                for (int res_idx_n = 0; res_idx_n < TN; res_idx_n++) {
                    c_interim[(thread_row_subtile * TM + res_idx_m) * N + thread_col_subtile * TN + res_idx_n] =
                        thread_results[(wm_idx * TM + res_idx_m) * (TN * n_subtiles) + wn_idx * TN + res_idx_n];
                }
            }
        }
    }
    __syncthreads();

}