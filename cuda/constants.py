WARPSIZE = 32
BM = 128
BN = 128
BK = 16
TM = 8
TN = 8
WN = 64
WM = 64
N_SUBTILES = 2
M_SUBTILES = (WM * WN) / (WARPSIZE * TM * TN * N_SUBTILES)
NUMBER_OF_THREADS = ((BM * BN) / (WN * WM)) * WARPSIZE
MSE_BM = 128
MSE_BN = 16
MSE_TM = 4
MSE_TN = 4
MSE_WM = 64
MSE_WN = 64
MSE_N_SUBTILES = 2
MSE_M_SUBTILES = (MSE_WM * MSE_WN) / (WARPSIZE * MSE_TM * MSE_TN * MSE_N_SUBTILES)
MSE_NUMBER_OF_THREADS = ((MSE_BM * MSE_BN) / (MSE_TM*MSE_TN))