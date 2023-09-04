all: libmatmul_benchmark.so libmatmul.so

libmatmul_benchmark.so:
	mkdir -p build && cd cuda && nvcc -shared -o ../build/libmatmul_benchmark.so ./kernels/matmul_2d.cu matmul_benchmark.cu kernel.cu -Xcompiler -fPIC -lcublas

libmatmul_benchmark_debug.so:
	mkdir -p build && cd cuda && nvcc -shared -o ../build/libmatmul_benchmark.so ./kernels/matmul_2d.cu matmul_benchmark.cu kernel.cu -Xcompiler -fPIC -lcublas

libmatmul.so:
	mkdir -p build && cd cuda && nvcc -shared -o ../build/libmatmul.so ./kernels/matmul_2d.cu kernel.cu -Xcompiler -fPIC -lcublas

libmse.so:
	mkdir -p build && cd cuda && nvcc -shared -o ../build/libmse.so ./kernels/mse.cu kernel.cu -Xcompiler -fPIC -lcublas

libkernels.so:
	mkdir -p build && cd cuda && nvcc -shared -o ../build/libkernels.so ./kernels/matmul_2d.cu ./kernels/mse.cu kernel.cu -Xcompiler -fPIC -lcublas

debug: libmatmul_benchmark_debug.so
	mkdir -p profile && /opt/nvidia/nsight-compute/2023.2.1/ncu -o ./profile/benchmark_matmul_profile python3 benchmark_matmul.py	

benchmark: libmatmul_benchmark.so
	python3 benchmark_matmul.py

matmul: libmatmul.so
	python3 benchmark.py

mse: libkernels.so
	python3 benchmark_mse.py

kernels: libkernels.so

test: libmatmul.so
	pytest -s ./test/test.py

clean:
	rm -rf build/ profile/