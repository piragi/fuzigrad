// Include guard to prevent double inclusion
#ifndef MATMUL_2D_H
#define MATMUL_2D_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
    // Function declarations
    __device__ void load_GMEM(float* a_local, float* b_local, float* a, float* b, const int N, const int K, int a_stride, int a_inner_row, int a_inner_col, int b_stride, int b_inner_row, int b_inner_col, int* flag_a);
    __device__ void load_SMEM(float* a_local, float* b_local, float* regM, float* regN, float* thread_results, const int thread_row_subtile, const int thread_col_subtile, const int wm_subtile, const int wn_subtile, const int m_subtiles, const int n_subtiles, const int warp_row, const int warp_col, int* flag_m, int* flag_n);
    __global__ void matmul_2d_tiling(float* a, float* b, float* c, const int M, const int N, const int K, int* flag, int* flag_m, int* flag_n, int* flag_a);
#ifdef __cplusplus
}
#endif
#endif // MATMUL_2D_H
