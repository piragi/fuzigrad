// Include guard to prevent double inclusion
#ifndef REDUCE
#define REDUCE

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif
// Function declarations
__device__ void load_GMEM(float* a, float* a_local, const int idx);
__device__ void load_SMEM(float* a_local, float* thread_value, const int idx);
__global__ void reduce(float* a);
#ifdef __cplusplus
}
#endif
#endif
