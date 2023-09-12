#include <cstdio>

#include "../constants.h"

__device__ void load_GMEM(float* a, float* a_local, const int idx) {
    int pos = idx * 4;
    reinterpret_cast<float4*>(&a_local[idx*4])[0] = reinterpret_cast<float4*>(&a[pos])[0]; 
}

__device__ void load_SMEM(float* a_local, float* thread_value, const int idx) {
    float4 tmp = reinterpret_cast<float4*>(&a_local[idx * 4])[0];
    *thread_value += tmp.w + tmp.x + tmp.y + tmp.z;
}

__device__ void shuffle_down_warps_reduce(float* thread_value) {
    unsigned mask = __ballot_sync(0xffffffff, 1);
    for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
        *thread_value += __shfl_down_sync(mask, *thread_value, offset);
    }
    // TODO: it is 128 for every thread? should not be
}

extern "C" __global__ void reduce(float* a, const int M) {
    __shared__ float values[WARPSIZE * 4];
    float thread_value = 0.0;

    const int idx = threadIdx.x;
    const int warpthread_idx = idx % WARPSIZE;
    const int warp_idx = idx / WARPSIZE;

    a += blockIdx.x * WARPSIZE * 4;

    load_GMEM(a, values, warpthread_idx);
    __syncthreads();
    load_SMEM(values, &thread_value, warpthread_idx);
    __syncthreads();
    shuffle_down_warps_reduce(&thread_value);

    if (warpthread_idx == 0) {
        printf("block %d, warp %d, value %f\n", blockIdx.x, warp_idx, thread_value);
        a[blockIdx.x] = thread_value;
    }
}