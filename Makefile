NVCC = nvcc
NVCC_FLAGS = -Xcompiler -fPIC -lcublas
DEBUG_FLAGS = -g -G

SRC_DIR = .
BUILD_DIR = $(SRC_DIR)/build
CUDA_DIR = $(SRC_DIR)/cuda
KERNELS_DIR = $(CUDA_DIR)/kernels
PROFILE_DIR = $(SRC_DIR)/profile/generated_profiles

# List of all the .cu files
CUDA_FILES = $(wildcard $(CUDA_DIR)/*.cu $(KERNELS_DIR)/*.cu)

# Name of the output file
OUT_FILE = $(BUILD_DIR)/libkernel.so
DEBUG_OUT_FILE = $(BUILD_DIR)/libkernel.so

all: $(OUT_FILE)

debug: $(DEBUG_OUT_FILE)

profile: $(OUT_FILE)
	mkdir -p $(PROFILE_DIR)
	/opt/nvidia/nsight-compute/2023.2.1/ncu -o $(PROFILE_DIR)/reduce_profile python3 ./profile/profile_reduce.py
	/opt/nvidia/nsight-compute/2023.2.1/ncu -o $(PROFILE_DIR)/mse_profile python3 ./profile/profile_mse.py
	/opt/nvidia/nsight-compute/2023.2.1/ncu -o $(PROFILE_DIR)/matmul_profile python3 ./profile/profile_matmul.py


$(OUT_FILE): $(CUDA_FILES)
	mkdir -p $(BUILD_DIR)
	$(NVCC) -shared -o $@ $(NVCC_FLAGS) $^

$(DEBUG_OUT_FILE): $(CUDA_FILES)
	rm -rf $(BUILD_DIR)/ $(PROFILE_DIR)/
	mkdir -p $(BUILD_DIR)
	$(NVCC) -shared -o $@ $(NVCC_FLAGS) $(DEBUG_FLAGS) $^

clean:
	rm -rf $(BUILD_DIR)/ $(PROFILE_DIR)/

.PHONY: all debug profile clean