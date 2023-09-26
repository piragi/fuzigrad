# Fuzigrad
Implementation of ML Framework for my educational purposes.
Inspirations: [karpathy/micrograd](https://github.com/karpathy/micrograd/) and [geohot/tinygrad](https://github.com/geohot/tinygrad/). Everything is handwritten down to the CUDA Kernels.

## Roadlist:
- **First trained network!**
    - doing basic MNIST classification
- **Accelerator**
    - Matmul with tiling with reference from [here](https://siboehm.com/articles/22/CUDA-MMM)
    - Basic operations in CUDA
    - Broadcasting
- **Basic Functions**
    - MSE
    - Reduce
    - Matmul
    - Sigmoid
    - improved SGD
    - DataLoader


## Install:
To compile the project:
```bash
make
```
To compile in debug mode:
```bash
make debug
```
To generate profiles for inspection with Nvidia Nsight:
```bash
make profile
```

## Benchmark:
Start benchmarking the custom CUDA kernels:
```bash
python3 benchmark.py
```

## Tests:
Testing is started via:
```bash
pytest test
```