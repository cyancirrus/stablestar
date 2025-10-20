cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(stablestar LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Bullet REQUIRED)
include_directories(${BULLET_INCLUDE_DIRS})

add_executable(stablestar
	src/main.cpp
	src/pendulum.cpp
)
target_link_libraries(stablestar ${BULLET_LIBRARIES})


# set_target_properties(stablestar PROPERTIES
# 	CUDA_ARCHITECTURES 75
# )
