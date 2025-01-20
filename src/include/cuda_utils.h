#pragma once

#define CUDA_CHECK(callstr) do { \
    cudaError_t error = callstr; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error in " << #callstr << " at " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)
