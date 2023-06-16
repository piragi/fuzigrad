import pyopencl as cl
import numpy as np

context = cl.create_some_context()
queue = cl.CommandQueue(context)
mf = cl.mem_flags

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

    print("matmul:")
    print(tensor1.value)
    print(tensor2.value)
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
    """).build()
    a_buf, b_buf = create_buffer( a, b)
    c_np = np.zeros((a.shape[0], b.shape[1])).astype(np.float32)
    c_buf = cl.Buffer(context, mf.WRITE_ONLY, c_np.nbytes)
    matmul.matmul2d(queue, c_np.shape, None, a_buf, b_buf, c_buf, np.int32(a.shape[1]), np.int32(b.shape[1]), np.int32(a.shape[0]))
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


