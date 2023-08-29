#pragma once

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8
#define WARPSIZE 32
#define WN 64
#define WM 64
#define NUMBER_OF_THREADS 128
#define N_SUBTILES 2
#define M_SUBTILES (WM * WN) / (WARPSIZE * TM * TN * N_SUBTILES)