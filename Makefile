all: libmatmul_benchmark.so

libmatmul_benchmark.so:
	cd cuda && nvcc -shared -o libmatmul_benchmark.so matmul_2d.cu matmul_benchmark.cu -Xcompiler -fPIC -lcublas

benchmark: libmatmul_benchmark.so
	python3 benchmark_matmul.py

clean:
	rm -f cuda/libmatmul_benchmark.so