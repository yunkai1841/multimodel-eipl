#include <iostream>
#include <gflags/gflags.h>

#include <random>

#include "cuda_utils.h"
#include "network.h"
#include "timer.h"

DEFINE_string(engine, "model.engine", "Path to TensorRT engine file");

const int IMAGE_SIZE = 3 * 64 * 64;
const int JOINT_SIZE = 8;
const int STATE_SIZE = 50;
const int PTS_SIZE = 2 * 5;

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << "Engine file: " << FLAGS_engine << std::endl;

    Network network(FLAGS_engine, IMAGE_SIZE, JOINT_SIZE, STATE_SIZE, PTS_SIZE);

    Timer timer;

    // random data for testing
    std::vector<float> image(IMAGE_SIZE);
    std::vector<float> joint(JOINT_SIZE);
    std::vector<float> state_h(STATE_SIZE);
    std::vector<float> state_c(STATE_SIZE);
    std::vector<float> output_image(IMAGE_SIZE);
    std::vector<float> output_joint(JOINT_SIZE);
    std::vector<float> output_state_h(STATE_SIZE);
    std::vector<float> output_state_c(STATE_SIZE);
    std::vector<float> output_ect_pts(PTS_SIZE);
    std::vector<float> output_dec_pts(PTS_SIZE);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    for (size_t i = 0; i < image.size(); i++)
    {
        image[i] = dis(gen);
    }

    for (size_t i = 0; i < joint.size(); i++)
    {
        joint[i] = dis(gen);
    }

    for (size_t i = 0; i < state_h.size(); i++)
    {
        state_h[i] = dis(gen);
    }

    for (size_t i = 0; i < state_c.size(); i++)
    {
        state_c[i] = dis(gen);
    }

    // allocate device memory
    float *d_image, *d_joint, *d_state_h, *d_state_c;
    CUDA_CHECK(cudaMalloc(&d_image, IMAGE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_joint, JOINT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_state_h, STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_state_c, STATE_SIZE * sizeof(float)));
    float *d_output_image, *d_output_joint, *d_output_state_h, *d_output_state_c, *d_output_ect_pts, *d_output_dec_pts;
    CUDA_CHECK(cudaMalloc(&d_output_image, IMAGE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_joint, JOINT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_state_h, STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_state_c, STATE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_ect_pts, PTS_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_dec_pts, PTS_SIZE * sizeof(float)));

    // copy data to device
    CUDA_CHECK(cudaMemcpy(d_image, image.data(), IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_joint, joint.data(), JOINT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state_h, state_h.data(), STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_state_c, state_c.data(), STATE_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // run inference
    timer.reset();
    network.allocate_bindings(d_image, d_joint, d_state_h, d_state_c,
                    d_output_image, d_output_joint, d_output_state_h, d_output_state_c,
                    d_output_ect_pts, d_output_dec_pts);
    network.forward(nullptr);
    timer.print_elapsed("Inference");

    // copy output back to host
    timer.reset();
    CUDA_CHECK(cudaMemcpy(output_image.data(), d_output_image, IMAGE_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output_joint.data(), d_output_joint, JOINT_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output_state_h.data(), d_output_state_h, STATE_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output_state_c.data(), d_output_state_c, STATE_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output_ect_pts.data(), d_output_ect_pts, PTS_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output_dec_pts.data(), d_output_dec_pts, PTS_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    timer.print_elapsed("Copy output");

    return 0;
}