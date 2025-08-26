cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(moonlight LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

add_executable(moonlight
	src/main.cpp
)

set_target_properties(moonlight PROPERTIES
	CUDA_ARCHITECTURES 75
)
