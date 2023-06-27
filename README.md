# Fuzigrad
Implementation of a autograd in python for educational purposes.

## Roadlist:
- **First trained network!**
    - Linear Layer (as simple as it gets)
    - adding them in a Forward structure
- **Accelerator**
    - Matmul with tiling
    - Load leafs into buffers at start, have a queue of operation and memory access
    - Broadcasting