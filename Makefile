all: libmatmul_benchmark.so libmatmul.so

libmatmul_benchmark.so:
	cd cuda && nvcc -shared -o ../build/libmatmul_benchmark.so ./kernels/matmul_2d.cu matmul_benchmark.cu kernel.cu -Xcompiler -fPIC -lcublas

libmatmul.so:
	cd cuda && nvcc -shared -o ../build/libmatmul.so ./kernels/matmul_2d.cu kernel.cu -Xcompiler -fPIC -lcublas

libkernels.so:
	cd cuda && nvcc -shared -o ../build/libkernels.so ./kernels/matmul_2d.cu ./kernels/mse.cu kernel.cu -Xcompiler -fPIC -lcublas

benchmark: libmatmul_benchmark.so
	python3 benchmark_matmul.py

matmul: libmatmul.so
	python3 benchmark.py

kernels: libkernels.so

test: libmatmul.so
	pytest -s ./test/test.py

clean:
	rm -f build/libmatmul_benchmark.so build/libmatmul.so