all: libmatmul_benchmark.so libmatmul.so

libmatmul_benchmark.so:
	cd cuda && nvcc -shared -o libmatmul_benchmark.so matmul_2d.cu matmul_benchmark.cu kernel.cu -Xcompiler -fPIC -lcublas

libmatmul.so:
	cd cuda && nvcc -shared -o libmatmul.so matmul_2d.cu kernel.cu -Xcompiler -fPIC -lcublas

benchmark: libmatmul_benchmark.so
	python3 benchmark_matmul.py

matmul: libmatmul.so
	python3 benchmark.py

test: libmatmul.so
	pytest -s test.py

clean:
	rm -f cuda/libmatmul_benchmark.so