// Include guard to prevent double inclusion
#ifndef MSE_2D_H
#define MSE_2D_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
    // Function declarations
    __device__ void load_GMEM(float* a, float* b, const int M, const int N, float* a_local, float* b_local, const int inner_row, const int inner_col, const int stride);
    __device__ void load_SMEM(float* a_local, float* b_local, const int M, const int N, float* thread_results);
    __global__ void mean_squared_error(float* a, float* b, float* thread_result, const int M, const int N);
#ifdef __cplusplus
}
#endif
#endif // MATMUL_2D_H

