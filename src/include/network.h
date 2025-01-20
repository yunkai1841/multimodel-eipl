#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>

#include "NvInfer.h"

#include "cuda_utils.h"
#include "logger.h"
#include "timer.h"

class Network
{
public:
    Network(std::string engine_path,
                     int image_size = 3 * 64 * 64,
                     int joint_size = 8,
                     int state_size = 50,
                     int pts_size = 2 * 5,
                     std::string name = "");
    ~Network() = default;

    void forward(cudaStream_t stream);
    void allocate_bindings(float *input_image, float *input_joint, float *input_state_h, float *input_state_c,
                           float *output_image, float *output_joint, float *output_state_h, float *output_state_c,
                           float *output_ect_pts, float *output_dec_pts);

    std::string name() const { return name_; }

private:
    void init_engine(std::string engine_path);
    void set_binding(const std::string &name, void *ptr);

    size_t image_size_ = 3 * 64 * 64;
    size_t joint_size_ = 8;
    size_t state_size_ = 50;
    size_t pts_size_ = 2 * 5;

    std::string name_;

    int total_networks_{1};
    nvinfer1::IExecutionContext *context_{nullptr};

    Timer timer_;
};
