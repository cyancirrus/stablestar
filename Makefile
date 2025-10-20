ARCH := sm_75

APPLICATION_NAME := stablestar
VERSION := 0.0.0

# Compiler 
# COMPILIER := nvcc
COMPILIER := nvcc

BULLET_INCLUDE := -I/usr/include/bullet \

LINKER_FLAGS := -lBulletDynamics \
				-lBulletCollision \
				-lLinearMath \
				-lGL \
				-lGLEW \
				-lglfw

COMPILIER_HOST_DEBUG_FLAGS := -arch=$(ARCH) \
			  -O1 \
			  -g -G \
			  -Xcompiler -fsanitize=address \
			  -Xcompiler -fsanitize=undefined \
			  -Xcompiler -fsanitize=leak \
			  ${BULLET_INCLUDE}

COMPILIER_CUDA_DEBUG_FLAGS := -arch=$(ARCH) \
			  -O1 \
			  -g -G \
			  ${BULLET_INCLUDE}

COMPILIER_RELEASE_FLAGS := -arch=$(ARCH) \
			  -O2 \
			  ${BULLET_INCLUDE}

# Targets
BUILD_DIR := target
RELEASE_DIR := build
# SRC := src/main.cu src/pipeline.cu
SRC := src/main.cpp src/pendulum.cpp
TARGET := $(BUILD_DIR)/$(APPLICATION_NAME)
RELEASE := $(RELEASE_DIR)/$(APPLICATION_NAME)-$(VERSION)

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(COMPILIER) $(COMPILIER_CUDA_DEBUG_FLAGS) $^ -o $@ $(LINKER_FLAGS)

run: $(TARGET)
	./$(TARGET)

release: $(SRC)
	mkdir -p $(RELEASE_DIR)
	$(COMPILIER) $(COMPILIER_RELEASE_FLAGS) $^ -o $(RELEASE)/$(APPLICATION_NAME)

memcheck: $(TARGET)
	compute-sanitizer --tool memcheck $(RELEASE)/$(APPLICATION_NAME)

racecheck: $(TARGET)
	compute-sanitizer --tool racecheck $(RELEASE)/$(APPLICATION_NAME)

diagnostics: release memcheck racecheck

clean:
	rm -f $(TARGET) $(RELEASE)/$(APPLICATION_NAME)
