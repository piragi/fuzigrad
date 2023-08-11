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
    # Add here any other device properties you are interested in

def create_buffer( a, b):
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    return a_buf, b_buf

def transpose(tensor):
    a = tensor.value.astype(np.float32)

    t = cl.Program(context, """
        __kernel void transpose(__global const float* a, __global float* b, const int shape0, const int shape1) {
            int row = get_global_id(0);
            int column = get_global_id(1);

            int idy = row + column * shape0;
            int idx = column + row * shape1;
            b[idy] = a[idx];
        }
    """).build()
    a_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
    transposed_np = np.zeros((tensor.shape[1], tensor.shape[0])).astype(np.float32)
    transposed_buf = cl.Buffer(context, mf.READ_ONLY, transposed_np.nbytes)
    t.transpose(queue, a.shape, None, a_buf, transposed_buf, np.int32(a.shape[0]), np.int32(a.shape[1]))
    cl.enqueue_copy(queue, transposed_np, transposed_buf)
    return transposed_np

def matmul( tensor1, tensor2):
    a = tensor1.value.astype(np.float32)
    b = tensor2.value.astype(np.float32)

    assert tensor1.shape[1] == tensor2.shape[0]

    # TODO: tiling instead of fetching every cell from global memory
    matmul = cl.Program(context, """
        __kernel void matmul2d(__global const float* a, __global const float* b, __global float* c, const int N, const int K, const int M) {
            int row = get_global_id(0);
            int column = get_global_id(1);

            float sum = 0.0f;
            for (int i = 0; i < N; i++) {
                sum += a[row * N + i] * b[i * K + column];
            }

            c[row * K + column] = sum;
        }

        __kernel void matmul(__global const float* a, __global const float* b, __global float* c, const int M, const int N, const int K) {
            // position in c
            const int x = get_group_id(0)
        }
    """).build()
    warp_size = matmul.matmul2d.get_work_group_info(
    cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
    device)
    print(f'Warp size: {warp_size}')
    a_buf, b_buf = create_buffer( a, b)
    c_np = np.zeros((a.shape[0], b.shape[1])).astype(np.float32)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, c_np.nbytes)
    start_time = time.time()
    matmul.matmul2d(queue, c_np.shape, None, a_buf, b_buf, c_buf, np.int32(a.shape[1]), np.int32(b.shape[1]), np.int32(a.shape[0]))
    matmul.matmul(queue, c_np.shape, (64, ), a_buf, b_buf, c_buf, np.int32(a.shape[1]), np.int32(b.shape[1]), np.int32(a.shape[0]))

    print(f'Matmul: {time.time() - start_time} seconds')

    cl.enqueue_copy(queue, c_np, c_buf)
    return c_np

def add( tensor1, tensor2):
    a = tensor1.value.astype(np.float32)
    b = tensor2.value.astype(np.float32)

    assert tensor1.shape == tensor2.shape

    sum = cl.Program(context, """
        __kernel void sum2d(__global const float* a, __global const float* b, __global float* c, const int size) {
            int row = get_global_id(0);
            int column = get_global_id(1);
            int idx = column + row * size;

            c[idx] = a[idx] + b[idx];
        }
    """).build()
    a_buf, b_buf = create_buffer( a, b)
    c_np = np.zeros_like(a).astype(np.float32)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, c_np.nbytes)
    sum.sum2d(queue, c_np.shape, None, a_buf, b_buf, c_buf, np.int32(a.shape[1]))
    cl.enqueue_copy(queue, c_np, c_buf)
    return c_np

def mul( tensor1, tensor2):
    if tensor2.shape == (1,):
        return mul_scalar(tensor1, tensor2)
    
def mul_scalar( tensor1, tensor2):
    a = tensor1.value.astype(np.float32)
    b = tensor2.value[0].astype(np.float32)

    mul = cl.Program(context, """
        __kernel void mul_scalar(__global const float* a, const float b, __global float* c, const int size) {
            int row = get_global_id(0);
            int column = get_global_id(1);
            int idx = column + row * size;

            c[idx] = a[idx] * b;
        }
    """).build()

    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    c_np = np.zeros_like(a).astype(np.float32)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, c_np.nbytes)
    mul.mul_scalar(queue, c_np.shape, None, a_buf, b, c_buf, np.int32(a.shape[1]))
    cl.enqueue_copy(queue, c_np, c_buf)
    return c_np

def relu(tensor):
    a = tensor.value.astype(np.float32)

    relu = cl.Program(context, """
        __kernel void relu(__global float* a, const int size) {
            int row = get_global_id(0);
            int column = get_global_id(1);

            int idx = column + row * size;
            if (a[idx] <= 0) {
                a[idx] = 0;
            }
        }
    """).build()
    a_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
    relu.relu(queue, a.shape, None, a_buf, np.int32(a.shape[1]))
    cl.enqueue_copy(queue, a, a_buf)
    return a

def relu_backwards(tensor):
    a = tensor.value.astype(np.float32)

    relu = cl.Program(context, """
        __kernel void relu(__global float* a, const int size) {
            int row = get_global_id(0);
            int column = get_global_id(1);

            int idx = column + row * size;
            if (a[idx] > 0) {
                a[idx] = 1;
            }
        }
    """).build()
    a_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=a)
    relu.relu(queue, a.shape, None, a_buf, np.int32(a.shape[1]))
    cl.enqueue_copy(queue, a, a_buf)
    return a

def is_even(n):
    return n > 0 and (n%2) == 0

def mse(tensor1, tensor2):
    start_time = time.time()
    a = tensor1.value.astype(np.float32)
    b = tensor2.value.astype(np.float32)

    mse = cl.Program(context, """
        __kernel void squared_difference(__global float* a, __global float* b, __global float* c, const int size) {
            int row = get_global_id(0);
            int column = get_global_id(1);
            int idx = column + row * size;

            c[idx] = pow(a[idx] - b[idx], 2);
        }

        __kernel void sum(__global float* a, const int size, const int num_elements) {
            int row = get_global_id(0);
            int column = get_global_id(1);
            int idx = column + row * size;
            
            a[idx] += a[idx + num_elements];
        }

        __kernel void sum_remainder(__global float* a, const int size, const int num_elements) {
            int row = get_global_id(0);
            int idx = row * size;

            a[idx] += a[idx + (num_elements*2)];
        }

        __kernel void sum_row_wise(__global float* a, const int size, const int num_elements) {
            int row = get_global_id(0);
            int idx = row * size;
            
            a[idx] += a[idx + num_elements*size];
        }
        
        __kernel void sum_row_wise_remainder(__global float* a, const int size, const int num_elements) {
            int row = get_global_id(0);
            int idx = row * size;
            
            a[0] += a[size*(num_elements*2)];
        }
    """).build()
    a_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, a.nbytes)
    #print(f'Move buffer a,b to device, create c through kernel: {time.time() - start_time} seconds')
    start_time = time.time()

    mse.squared_difference(queue, a.shape, None, a_buf, b_buf, c_buf, np.int32(a.shape[1]))
  
    #print(f'Squared difference: {time.time() - start_time} seconds')
    start_time = time.time()

    even = is_even(a.shape[1])
    num_elements = a.shape[1] // 2
    while (num_elements != 0):
        mse.sum(queue, (a.shape[0], num_elements), None, c_buf, np.int32(a.shape[1]), np.int32(num_elements))
        if not even:
            mse.sum_remainder(queue, (a.shape[0],), None, c_buf, np.int32(a.shape[1]), np.int32(num_elements))
        even = is_even(num_elements)
        num_elements = num_elements//2
    
    #print(f'Row-wise sum: {time.time() - start_time} seconds')
    start_time = time.time()

    even = is_even(a.shape[0])
    num_elements = a.shape[0] // 2
    while (num_elements != 0):
        mse.sum_row_wise(queue, (num_elements,), None, c_buf, np.int32(a.shape[1]), np.int32(num_elements))
        if not even:
            mse.sum_row_wise_remainder(queue, (1,), None, c_buf, np.int32(a.shape[1]), np.int32(num_elements))
        even = is_even(num_elements)
        num_elements = num_elements//2

    #print(f'Column-wise sum: {time.time() - start_time} seconds')
    start_time = time.time()

    result = np.zeros(1, dtype=np.float32)
    cl.enqueue_copy(queue, result, c_buf, src_offset=0)
    
    #print(f'Transfer c to scalar: {time.time() - start_time} seconds')
    start_time = time.time()
    
    return result/a.size


