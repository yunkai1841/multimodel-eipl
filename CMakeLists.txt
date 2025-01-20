cmake_minimum_required(VERSION 3.10)

project(MultiTaskBenchmark)

# Set the source directory
set(SOURCE_DIR src)

set(TEST_DIR test)

# Set include directories
include_directories(${SOURCE_DIR}/include)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/aarch64-linux-gnu
    PATH_SUFFIXES tensorrt)

# Find gflags
find_package(gflags REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)

include_directories(${TENSORRT_INCLUDE_DIR} ${gflags_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

# network library
add_library(network SHARED ${SOURCE_DIR}/network.cc)
target_link_libraries(network ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvparsers)

# main
add_executable(multitask_benchmark ${SOURCE_DIR}/main.cc)
target_link_libraries(multitask_benchmark gflags ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvparsers network)

# test
add_executable(test_network ${TEST_DIR}/test_network.cc)
target_link_libraries(test_network gflags ${CUDA_LIBRARIES} nvinfer nvinfer_plugin nvparsers network)

