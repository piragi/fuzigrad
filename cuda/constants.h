#pragma once

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8
#define WARPSIZE 32
#define WN 64
#define WM 64
#define N_SUBTILES 2
#define M_SUBTILES (WM * WN) / (WARPSIZE * TM * TN * N_SUBTILES)

// one grid per block
// one grid calculates BM * BN results
// (BM * BN) / (WN * WM) warptiles
// WARPSIZE threads per warp and warptile
// therefore:
#define NUMBER_OF_THREADS ((BM * BN) / (WN * WM)) * WARPSIZE


// ----- MSE -----
#define MSE_TM 4
#define MSE_TN 4