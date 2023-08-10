#include <assert.h>
#include "constants.h"

extern "C" __global__ void matmul_2d_tiling(float* a, float* b, float* c, const int M, const int N, const int K) {
    const int number_of_threads = blockDim.x * blockDim.y;
    const int tile_size = BM * BK;


    const int c_row = blockIdx.y;
    const int c_col = blockIdx.x;

    const int thread_row = threadIdx.y;
    const int thread_col = threadIdx.x;
    const int idx = thread_row * blockDim.x + thread_col;

    __shared__ float a_local[BM * BK];
    __shared__ float b_local[BK * BN];

    a += c_row * BM * K;
    b += c_col * BN;
    c += c_row * BM * N + c_col * BN;

    float thread_results[TM * TN] = {0};
    float regM[TM] = {0};
    float regN[TN] = {0};

    const int a_inner_col = idx % (BK / 4);
    const int a_inner_row = idx / (BK / 4);
    const int a_stride = (number_of_threads * 4) / BK;

    const int b_inner_col = idx % (BN / 4);
    const int b_inner_row = idx / (BN / 4);
    // in the sgemm version its done differently, i dont see a reason why tho
    const int b_stride = (number_of_threads * 4) / BN;

    // to guarantue float4 loads
    //assert(tile_size / number_of_threads % 4 == 0);
    
    for (int block_idx = 0; block_idx < K; block_idx += BK) {
        // load 4 floats into local memory (L1 SMEM) 
        // number of elements per tile / number of threads need to be dividable by 4
        // (BK * BM) / (number_of_threads) % 4 == 0 
        // meaning 1024 / 256 = 4
        // to achieve a BK != TM we need to load multiple float4s
        // meaning one thread loads float4 from row x and row x + offset (for 2048 elements with 256 threads)
        for (int offset=0; offset < BM; offset+=a_stride) {
            float4 tmp = reinterpret_cast<float4 *>(&a[(a_inner_row + offset) * K + a_inner_col * 4])[0];
            a_local[(a_inner_col * 4 + 0) * BM + a_inner_row + offset] = tmp.x;  
            a_local[(a_inner_col * 4 + 1) * BM + a_inner_row + offset] = tmp.y;
            a_local[(a_inner_col * 4 + 2) * BM + a_inner_row + offset] = tmp.z;
            a_local[(a_inner_col * 4 + 3) * BM + a_inner_row + offset] = tmp.w;
        }
        for (int offset=0; offset < BK; offset+=b_stride) {
            reinterpret_cast<float4 *>(&b_local[(b_inner_row + offset) * BN + b_inner_col * 4])[0] = reinterpret_cast<float4 *>(&b[(b_inner_row + offset) * N + b_inner_col * 4])[0];
        } 
        __syncthreads();

        // move the tile sideways
        a += BK;
        // move the tile downwards
        b += BK * N;

        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            for (int a_idx = 0; a_idx < TM; a_idx++) {
                regM[a_idx] = a_local[(dot_idx * BM) + thread_row * TM + a_idx];
            }
            for (int b_idx = 0; b_idx < TN; b_idx++) {
                regN[b_idx] = b_local[(dot_idx * BN) + thread_col * TN + b_idx];
            }
            
            for (int res_idx_m = 0; res_idx_m < TM; res_idx_m++) {
                for (int res_idx_n = 0; res_idx_n < TN; res_idx_n++) {
                    thread_results[res_idx_m * TN + res_idx_n] += regM[res_idx_m] * regN[res_idx_n];
                }
            }
        }
        __syncthreads();
    }
    
    for (int res_idx_m = 0; res_idx_m < TM; res_idx_m++) {
        for (int res_idx_n = 0; res_idx_n < TN; res_idx_n++) {
            c[(thread_row * TM + res_idx_m) * N + thread_col * TN + res_idx_n] = thread_results[res_idx_m * TN + res_idx_n];
        }
    }
}