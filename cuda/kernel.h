#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

    void matmul(float* a, float* b, float* c, const int M, const int N, const int K);
    void mse(float* a, float* b, float* c, const int M, const int N);

#ifdef __cplusplus
}
#endif

#endif // KERNEL_H
