#pragma once

#define WARPSIZE 32

// ---- GEMM ----
#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8
#define WN 64
#define WM 64
#define N_SUBTILES 2
#define M_SUBTILES ((WM * WN) / (WARPSIZE * TM * TN * N_SUBTILES))

// one grid per block
// one grid calculates BM * BN results
// (BM * BN) / (WN * WM) warptiles
// WARPSIZE threads per warp and warptile
// therefore:
#define NUMBER_OF_THREADS ((BM * BN) / (WN * WM)) * WARPSIZE

// ---- MSE ----
#define MSE_BM 64
#define MSE_BN 64
#define MSE_TM 4
#define MSE_TN 4
#define MSE_WM 32
#define MSE_WN 32
#define MSE_N_SUBTILES 2
#define MSE_M_SUBTILES ((MSE_WM * MSE_WN) / (WARPSIZE * MSE_TM * MSE_TN * MSE_N_SUBTILES))

#define MSE_NUMBER_OF_THREADS ((MSE_BM * MSE_BN) / (MSE_WM * MSE_WN)) * WARPSIZE

// ---- Reduce ----
#define REDUCE_BM 128
#define REDUCE_NUMBER_OF_THREADS (REDUCE_BM / 4)