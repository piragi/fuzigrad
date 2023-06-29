import pyopencl as cl
import numpy as np
import time

context = cl.create_some_context()
device = context.devices[0]
queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)
mf = cl.mem_flags

# Now, 'devices' is a list of devices available in the context. 
# You can print information about each device:
print(f'Device name: {device.name}')
print(f'Device type: {cl.device_type.to_string(device.type)}')
print(f'Device memory: {device.global_mem_size//1024//1024} MB')
print(f'Max compute units: {device.max_compute_units}')
print(f'Max work group size: {device.max_work_group_size}')
print(f"The local memory size of the device is {device.get_info(cl.device_info.LOCAL_MEM_SIZE)} KB")


max_wg_size = device.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)

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
        #define BM 32
        #define BN 32
        #define BK 32
        #define TM 2

        __kernel void matmul_local(__global const float* a, __global const float* b, __global float* c, const int M, const int N, const int K) {
            const int c_row = get_group_id(1);
            const int c_col = get_group_id(0);

            // this should be a multiple of 32?
            const int thread_row = get_local_id(1);
            const int thread_col = get_local_id(0);

            __local float a_local[BM * BK];
            __local float b_local[BK * BN];

            a += c_row * BM * K;
            b += c_col * BN;
            c += c_row * BM * N + c_col * BN;

            // cache for each thread
            float thread_result[TM];
            for (int i = 0; i < TM; i++) {
                threadResults[i] = 0.0f;
            }

            if (thread_row < M && thread_col < N) {
                for (int block_idx=0; block_idx < K; block_idx += BK) {
                    // load data into local memory    
                    a_local[thread_row * BK + thread_col] = a[thread_row * K + thread_col];
                    b_local[thread_row * BN + thread_col] = b[thread_row * N + thread_col];
                    // synch the load
                    barrier(CLK_LOCAL_MEM_FENCE);

                    // advance pointers
                    a += BK;
                    b += BK * N;

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