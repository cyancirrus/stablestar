ARCH := sm_75

APPLICATION_NAME := stablestar
VERSION := 0.0.0

# Compiler 
# COMPILIER := nvcc
COMPILIER := nvcc

INCLUDE_FLAGS := -I/usr/include/bullet \
				  -I/usr/include/stb \

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
			  ${INCLUDE_FLAGS}

COMPILIER_CUDA_DEBUG_FLAGS := -arch=$(ARCH) \
			  -O1 \
			  -g -G \
			  ${INCLUDE_FLAGS}

COMPILIER_RELEASE_FLAGS := -arch=$(ARCH) \
			  -O2 \
			  ${INCLUDE_FLAGS}

# Targets
BUILD_DIR := target
RELEASE_DIR := build
SRC := src/main.cpp src/cart_pendulum.cpp
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
