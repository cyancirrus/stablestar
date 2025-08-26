ARCH := sm_75

APPLICATION_NAME := moonlight
VERSION := 0.0.0

# Compiler 
NVCC := nvcc

NVCC_HOST_DEBUG_FLAGS := -arch=$(ARCH) \
			  -O1 \
			  -g -G \
			  -Xcompiler -fsanitize=address \
			  -Xcompiler -fsanitize=undefined \
			  -Xcompiler -fsanitize=leak \
			  -Wno-deprecated-gpu-targets

NVCC_CUDA_DEBUG_FLAGS := -arch=$(ARCH) \
			  -O1 \
			  -g -G \
			  -Wno-deprecated-gpu-targets

NVCC_RELEASE_FLAGS := -arch=$(ARCH) \
			  -O2 \
			  -Wno-deprecated-gpu-targets \

# Targets
BUILD_DIR := target
RELEASE_DIR := build
# SRC := src/main.cu src/pipeline.cu
SRC := src/main.cpp
TARGET := $(BUILD_DIR)/$(APPLICATION_NAME)
RELEASE := $(RELEASE_DIR)/$(APPLICATION_NAME)-$(VERSION)

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_CUDA_DEBUG_FLAGS) $^ -o $@

run: $(TARGET)
	./$(TARGET)

release: $(SRC)
	mkdir -p $(RELEASE_DIR)
	$(NVCC) $(NVCC_RELEASE_FLAGS) $^ -o $(RELEASE)/$(APPLICATION_NAME)

memcheck: $(TARGET)
	compute-sanitizer --tool memcheck $(RELEASE)/$(APPLICATION_NAME)

racecheck: $(TARGET)
	compute-sanitizer --tool racecheck $(RELEASE)/$(APPLICATION_NAME)

diagnostics: release memcheck racecheck

clean:
	rm -f $(TARGET) $(RELEASE)/$(APPLICATION_NAME)
