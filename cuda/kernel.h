#pragma once
#include <cuda_runtime.h>

void matmul(float* a, float* b, float* c, const int M, const int N, const int K);