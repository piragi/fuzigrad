import pyopencl as cl
import numpy as np
import time

context = cl.create_some_context()
device = context.devices[0]
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
mf = cl.mem_flags
max_wg_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

benchmark = False

if benchmark:
    runtime = 100
else:
    runtime = 1

def matmul_2d_blocktiling(tensor1, tensor2):
    a = tensor1.value.astype(np.float32)
    b = tensor2.value.astype(np.float32)

    assert tensor1.shape[1] == tensor2.shape[0]

    M = np.int32(a.shape[0])
    N = np.int32(b.shape[1])
    K = np.int32(a.shape[1])

    matmul = cl.Program(context, """
        // has to match work_group_size
        #define BM 64
        #define BN 64
        #define BK 8
        #define TM 8
        #define TN 8

        __kernel void matmul_2d_tiling(__global const float* a, __global const float* b, __global float* c, const int M, const int N, const int K) {
            const int c_row = get_group_id(1);
            const int c_col = get_group_id(0);

            const int thread_row = get_local_id(1);
            const int thread_col = get_local_id(0);

            const int idx = thread_row * BM + thread_col;

            const int totalResultBlocktile = BM * BN;
            const int threadsPerBlocktile = totalResultBlocktile / (TM * TN);

            __local float a_local[BM * BK];
            __local float b_local[BK * BN];

            a += c_row * BM * K;
            b += c_col * BN;
            c += c_row * BM * N + c_col * BN;

            // cache for each result
            float thread_results[TM * TN];
            for (int i = 0; i < (TM * TN); i++) {
                thread_results[i] = 0.0f;
            }

            // cache for row of local_a
            float regM[TM];
            for (int i = 0; i < TM; i++) {
                regM[i] = 0.0f;
            }

            // cache for column of local_b
            float regN[TN];
            for (int i = 0; i < TN; i++) {
                regN[i] = 0.0f;
            }

            const int a_inner_col = idx % BK;
            const int a_inner_row = idx / BK;
            const int a_stride = threadsPerBlocktile / BK;

            const int b_inner_col = idx % BN;
            const int b_inner_row = idx / BN;
            const int b_stride = threadsPerBlocktile / BN;

            for (int block_idx=0; block_idx < K; block_idx += BK) {
                // load data into local memory 
                for (int load_offset=0; load_offset < BM; load_offset += a_stride) {
                    a_local[(a_inner_row + load_offset) * BK + a_inner_col] = a[(a_inner_row + load_offset) * K + a_inner_col];
                }
                for (int load_offset=0; load_offset < BK; load_offset += b_stride) {
                    b_local[(b_inner_row + load_offset) * BN + b_inner_col] = b[(b_inner_row + load_offset) * N + b_inner_col];
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                // advance pointers
                a += BK;
                b += BK * N;

                for (int dot_idx=0; dot_idx < BK; dot_idx++) {
                    for (int a_idx=0; a_idx < TM; a_idx++) {
                        regM[a_idx] = a_local[(thread_row * TM + a_idx) * BK + dot_idx];
                    }
                    for (int b_idx=0; b_idx < TN; b_idx++) {
                        regN[b_idx] = b_local[(dot_idx * BN) + thread_col * TN + b_idx];
                    }
                    
                    // arithmetics -> add to results
                    for (int res_idx_m=0; res_idx_m < TM; res_idx_m++) {
                        for (int res_idx_n=0; res_idx_n < TN; res_idx_n++) {
                            thread_results[res_idx_m * TN + res_idx_n] += regM[res_idx_m] * regN[res_idx_n];
                        }
                    }
                } 
                // prevent reloading cache before update of c 
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            for (int res_idx_m=0; res_idx_m < TM; res_idx_m++) {
                for (int res_idx_n=0; res_idx_n < TN; res_idx_n++) {
                    c[(thread_row * TM + res_idx_m) * N + thread_col * TN + res_idx_n] = thread_results[res_idx_m * TN + res_idx_n];
                }
            }
        }
    """).build()

    BM = 64
    BN = 64
    BK = 8
    TM = 8
    TN = 8

    # NOTE: Check the dimensions of all the matrix
    warp_size = matmul.matmul_2d_tiling.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device)
    local_work_size = (BM // TM, BN // TN)
    global_work_size = ((M + (BM - 1)) // BM * BM // TM), ((N + (BN - 1)) // BN * BN // TN)

    #print(f'local work size: {local_work_size}')
    #print(f'global work size: {global_work_size}')
    
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_np = np.zeros((a.shape[0], b.shape[1])).astype(np.float32)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, c_np.nbytes)
    
    mean_elapsed = []
    for _ in range(runtime):
        kernel = matmul.matmul_2d_tiling
        kernel.set_args(a_buf, b_buf, c_buf, M, N, K)
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size)        
        event.wait()
        mean_elapsed.append(1e-9 * (event.profile.end - event.profile.start))
    print(f'2-d tiling = {np.mean(mean_elapsed)}')

    cl.enqueue_copy(queue, c_np, c_buf)
    return c_np

def matmul_1d_blocktiling(tensor1, tensor2):
    a = tensor1.value.astype(np.float32)
    b = tensor2.value.astype(np.float32)

    assert tensor1.shape[1] == tensor2.shape[0]

    M = np.int32(a.shape[0])
    N = np.int32(b.shape[1])
    K = np.int32(a.shape[1])

    matmul = cl.Program(context, """
        // has to match work_group_size
        #define LSIZE 8
        #define BM 64
        #define BN 64
        #define BK 8
        #define TM 8

        __kernel void matmul_1d_tiling(__global const float* a, __global const float* b, __global float* c, const int M, const int N, const int K) {
            const int c_row = get_group_id(1);
            const int c_col = get_group_id(0);

            const int thread_row = get_local_id(1);
            const int thread_col = get_local_id(0);

            const int idx = thread_row * BM + thread_col;

            __local float a_local[BM * BK];
            __local float b_local[BK * BN];

            a += c_row * BM * K;
            b += c_col * BN;
            c += c_row * BM * N + c_col * BN;

            // cache for each thread
            float thread_results[TM];
            for (int i = 0; i < TM; i++) {
                thread_results[i] = 0.0f;
            }

            const int a_inner_col = idx % BK;
            const int a_inner_row = idx / BK;
            const int b_inner_col = idx % BN;
            const int b_inner_row = idx / BN;

            if (c_row < M && c_col < N) {
                for (int block_idx=0; block_idx < K; block_idx += BK) {
                    // load data into local memory    
                    a_local[a_inner_row * BK + a_inner_col] = a[a_inner_row * K + a_inner_col];
                    b_local[b_inner_row * BN + b_inner_col] = b[b_inner_row * N + b_inner_col];
                    barrier(CLK_LOCAL_MEM_FENCE);

                    // advance pointers
                    a += BK;
                    b += BK * N;

                    for (int dot_idx=0; dot_idx < BK; dot_idx++) {
                        const float b_tmp = b_local[dot_idx * BN + thread_col];
                        for (int results_idx=0; results_idx < TM; results_idx++) {
                            thread_results[results_idx] += a_local[(thread_row * TM + results_idx) * BK + dot_idx] * b_tmp;
                        }
                    } 
                    // prevent reloading cache before update of c 
                    barrier(CLK_LOCAL_MEM_FENCE);
                }

                for (int results_idx=0; results_idx < TM; results_idx++) {
                    c[(thread_row * TM + results_idx) * N + thread_col] = thread_results[results_idx];
                }
            }
        }
    """).build()

    BM = 64
    BN = 64
    BK = 8
    TM = 8

    # NOTE: Check the dimensions of all the matrix
    warp_size = matmul.matmul_1d_tiling.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device)
    local_work_size = (BM, TM)
    global_work_size = ((M + (BM - 1)) // BM * BM), ((N + (BN - 1)) // BN * BN // TM)

    #print(f'local work size: {local_work_size}')
    #print(f'global work size: {global_work_size}')
    
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_np = np.zeros((a.shape[0], b.shape[1])).astype(np.float32)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, c_np.nbytes)
    
    mean_elapsed = []
    for _ in range(runtime):
        kernel = matmul.matmul_1d_tiling
        kernel.set_args(a_buf, b_buf, c_buf, M, N, K)
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size)        
        event.wait()
        mean_elapsed.append(1e-9 * (event.profile.end - event.profile.start))
    print(f'1-d tiling = {np.mean(mean_elapsed)}')

    cl.enqueue_copy(queue, c_np, c_buf)
    return c_np

def matmul_local(tensor1, tensor2):
    a = tensor1.value.astype(np.float32)
    b = tensor2.value.astype(np.float32)

    assert tensor1.shape[1] == tensor2.shape[0]

    M = np.int32(a.shape[0])
    N = np.int32(b.shape[1])
    K = np.int32(a.shape[1])

    matmul = cl.Program(context, """
        // has to match work_group_size
        #define LSIZE 8

        __kernel void matmul_local(__global const float* a, __global const float* b, __global float* c, const int M, const int N, const int K) {
            // position in c - switching up thread_row and thread_col results in
            // 2x slowdown!
            const int c_row = get_group_id(1);
            const int c_col = get_group_id(0); 
            const int thread_row = get_local_id(1);
            const int thread_col = get_local_id(0);

            __local float a_local[LSIZE * LSIZE];
            __local float b_local[LSIZE * LSIZE];

            a += c_row * LSIZE * K;
            b += c_col * LSIZE;
            c += c_row * LSIZE * N + c_col * LSIZE;

            if (thread_row < M && thread_col < N) {
                float sum = 0.0f;
                for (int block_idx=0; block_idx < K; block_idx += LSIZE) {
                    // load data into local memory    
                    a_local[thread_row * LSIZE + thread_col] = a[thread_row * K + thread_col];
                    b_local[thread_row * LSIZE + thread_col] = b[thread_row * N + thread_col];
                    // synch the load
                    barrier(CLK_LOCAL_MEM_FENCE);

                    // advance pointers
                    a += LSIZE;
                    b += LSIZE * N;

                    for (int i=0; i < LSIZE; i++) {
                        sum += a_local[thread_row * LSIZE + i] * b_local[i * LSIZE + thread_col];
                    }

                    // prevent reloading cache before update of c 
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                c[thread_row * N + thread_col] = sum;
            }
        }
    """).build()

    warp_size = matmul.matmul_local.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device)
    work_group_size = 8
    global_work_size = ((M + (work_group_size - 1)) // work_group_size * work_group_size), ((N + (work_group_size - 1)) // work_group_size * work_group_size)
    
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_np = np.zeros((a.shape[0], b.shape[1])).astype(np.float32)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, c_np.nbytes)
    
    mean_elapsed = []
    for _ in range(100):
        kernel = matmul.matmul_local
        kernel.set_args(a_buf, b_buf, c_buf, M, N, K)
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, (work_group_size, work_group_size))        
        event.wait()
        mean_elapsed.append(1e-9 * (event.profile.end - event.profile.start))
    print(f'Local Memory = {np.mean(mean_elapsed)}')

    cl.enqueue_copy(queue, c_np, c_buf)
    return c_np


def matmul( tensor1, tensor2):
    a = tensor1.value.astype(np.float32)
    b = tensor2.value.astype(np.float32)

    assert tensor1.shape[1] == tensor2.shape[0]

    M = np.int32(a.shape[0])
    N = np.int32(b.shape[1])
    K = np.int32(a.shape[1])

    # TODO: tiling instead of fetching every cell from global memory
    matmul = cl.Program(context, """
        // has to match work_group_size
        #define LSIZE 16

        __kernel void matmul_coalesced(__global const float* a, __global const float* b, __global float* c, const int M, const int N, const int K) {
            // position in c
            const int idx = get_local_id(1) + get_local_id(0) * get_local_size(0);
            const int x = get_group_id(0) * get_local_size(0) + (idx / get_local_size(0));
            const int y = get_group_id(1) * get_local_size(1) + (idx % get_local_size(1));

            if (x < M && y < N) {
                float sum = 0.0f;
                for (int i=0; i < K; i++) {
                    sum += a[x * K + i] * b[y + N * i];
                }
                c[x * N + y] = sum;
            }
        }
    """).build()

    warp_size = matmul.matmul_coalesced.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device)
    work_group_size = 16
    global_work_size = ((M + (work_group_size - 1)) // work_group_size * work_group_size), ((N + (work_group_size - 1)) // work_group_size * work_group_size)
    
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_np = np.zeros((a.shape[0], b.shape[1])).astype(np.float32)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, c_np.nbytes)
    
    mean_elapsed = []
    for _ in range(100):
        kernel = matmul.matmul_coalesced
        kernel.set_args(a_buf, b_buf, c_buf, M, N, K)
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, (work_group_size, work_group_size))        
        event.wait()
        mean_elapsed.append(1e-9 * (event.profile.end - event.profile.start))
    print(f'Coalesced = {np.mean(mean_elapsed)}')

    cl.enqueue_copy(queue, c_np, c_buf)
    return c_np

def matmul_normal(tensor1, tensor2):
    a = tensor1.value.astype(np.float32)
    b = tensor2.value.astype(np.float32)

    assert tensor1.shape[1] == tensor2.shape[0]

    M = np.int32(a.shape[0])
    N = np.int32(b.shape[1])
    K = np.int32(a.shape[1])

    # TODO: tiling instead of fetching every cell from global memory
    matmul = cl.Program(context, """
        // has to match work_group_size
        #define LSIZE 16

        __kernel void matmul(__global const float* a, __global const float* b, __global float* c, const int M, const int N, const int K) {
            // position in c
            const int x = get_group_id(0) * get_local_size(0) + get_local_id(0);
            const int y = get_group_id(1) * get_local_size(1) + get_local_id(1);

            if (x < M && y < N) {
                float sum = 0.0f;
                for (int i=0; i < K; i++) {
                    sum += a[x * K + i] * b[y + N * i];
                }
                c[x * N + y] = sum;
            }
        }
    """).build()

    warp_size = matmul.matmul.get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, device)
    work_group_size = 16
    global_work_size = ((M + (work_group_size - 1)) // work_group_size * work_group_size), ((N + (work_group_size - 1)) // work_group_size * work_group_size)

    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_np = np.zeros((a.shape[0], b.shape[1])).astype(np.float32)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, c_np.nbytes)

    mean_elapsed = []
    for _ in range(100):
        kernel = matmul.matmul
        kernel.set_args(a_buf, b_buf, c_buf, M, N, K)
        event = cl.enqueue_nd_range_kernel(queue, kernel, global_work_size, (work_group_size, work_group_size))        
        event.wait()
        mean_elapsed.append(1e-9 * (event.profile.end - event.profile.start))
    print(f'Normal = {np.mean(mean_elapsed)}')

    cl.enqueue_copy(queue, c_np, c_buf)
    return c_np