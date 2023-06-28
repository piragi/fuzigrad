import pyopencl as cl
import numpy as np
import time

context = cl.create_some_context()
device = context.devices[0]
queue = cl.CommandQueue(context)
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
            const int idx = get_local_id(0) + get_local_id(1) * get_local_size(0);
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
    
    start = time.time()
    for _ in range(100):
        matmul.matmul_coalesced(queue, global_work_size, (work_group_size, work_group_size), a_buf, b_buf, c_buf, M, N, K)
    print(f'Coalesced = {time.time() - start}')

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

    start = time.time()
    for _ in range(100):
        matmul.matmul(queue, global_work_size, (work_group_size, work_group_size), a_buf, b_buf, c_buf, M, N, K)
    print(f'Normal = {time.time() - start}')


    cl.enqueue_copy(queue, c_np, c_buf)
    return c_np