#include <cstdio>

#include "../constants.h"

__device__ void load_GMEM(float* a, float* a_local, const int idx) {
    reinterpret_cast<float4*>(&a_local[idx * 4])[0] = reinterpret_cast<float4*>(&a[idx * 4])[0];
}

__device__ void load_SMEM(float* a_local, float* thread_value, const int idx) {
    float4 tmp = reinterpret_cast<float4*>(&a_local[idx * 4])[0];
    *thread_value += tmp.w + tmp.x + tmp.y + tmp.z;
}

__device__ void shuffle_down_warps_reduce(float* reg_tile) {
    unsigned mask = __ballot_sync(0xffffffff, 1);
    for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
        *reg_tile += __shfl_down_sync(mask, *reg_tile, offset);
    }
}

extern "C" __global__ void reduce(float* a) {
    __shared__ float* values;
    float thread_value = 0.0;

    const int idx = threadIdx.x;
    const int warpthread_idx = idx % WARPSIZE;

    a += blockIdx.x * gridDim.x + blockIdx.y;
    printf("thread_id: %d, warp_id: %d\n", idx, warpthread_idx);
    load_GMEM(a, values, idx);
    __syncthreads();
    load_SMEM(values, &thread_value, idx);
    __syncthreads();
    shuffle_down_warps_reduce(&thread_value);
    if (idx == 0) {
        printf("this is: %f", thread_value);
        a[0] = thread_value;
    }
    __syncthreads();
}