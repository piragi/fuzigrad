#include <cstdio>

#include "../constants.h"

__device__ void load_GMEM(float* a, float* a_local, const int idx) {
    int global_pos = blockIdx.x * WARPSIZE * 4;
    int local_pos = idx * 4;
    reinterpret_cast<float4*>(&a_local[local_pos])[0] = reinterpret_cast<float4*>(&a[global_pos+local_pos])[0]; 
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

// start block with multiple warps
// do warp reduce
// store result in shared memory
// do shared memory reduce
// store result in global memory
// repeat until only one block left
extern "C" __global__ void reduce_warps(float* a, const int M, float* result) {
    __shared__ float values[WARPSIZE * 4];
    float thread_value = 0.0;

    const int idx = threadIdx.x;
    const int warpthread_idx = idx % WARPSIZE;
    const int warp_idx = idx / WARPSIZE;
    const int number_of_warps = REDUCE_NUMBER_OF_THREADS / WARPSIZE;


    load_GMEM(a, values, warpthread_idx);
    __syncthreads();
    load_SMEM(values, &thread_value, warpthread_idx);
    __syncthreads();
    shuffle_down_warps_reduce(&thread_value);

    if (warpthread_idx == 0) {
        int blockId = blockIdx.x;
        int blockDi = gridDim.x;
        int pos = blockId * number_of_warps + warp_idx;
        printf("(pos: %d) block %d, warp %d, value %f\n", pos, blockIdx.x, warp_idx, thread_value);
        result[pos] = thread_value;
    }
}