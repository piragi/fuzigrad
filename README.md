# Fuzigrad
Implementation of ML Framework for educational purposes.
Inspirations: [karpathy/micrograd](https://github.com/karpathy/micrograd/) and [geohot/tinygrad](https://github.com/geohot/tinygrad/).

## Roadlist:
- **First trained network!**
    - adding them in a Forward structure (aka. model = Model())
- **Accelerator**
    - Matmul with tiling with reference from [here](https://siboehm.com/articles/22/CUDA-MMM)
    - Load leafs into buffers at start, have a queue of operation and memory access
    - Broadcasting
- **Basic Functions**
    - Sigmoid
    - improved SGD
    - DataLoader


## Install:
To compile the project:
```bash
make
```
To compile and start the benchmark:
```bash
make benchmark
```