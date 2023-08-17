NVCC = nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

OBJS = runner.o matmul_2d.o
TARGET = runner

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) -o $(TARGET) -lcublas

runner.o: runner.cu matmul_2d.h
	$(NVCC) $(NVCC_FLAGS) -c runner.cu

matmul_2d.o: matmul_2d.cu matmul_2d.h
	$(NVCC) $(NVCC_FLAGS) -c matmul_2d.cu

clean:
	rm -f $(OBJS) $(TARGET)
