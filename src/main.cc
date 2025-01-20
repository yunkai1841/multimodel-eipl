#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <memory>

#include <cuda_runtime.h>

#include "NvInfer.h"

#include "cuda_utils.h"
#include "network.h"
#include "timer.h"
#include <gflags/gflags.h>

DEFINE_string(engine, "model.engine", "Path to TensorRT engine file");
DEFINE_int32(total_networks, 1, "Number of networks to run");
DEFINE_int32(loops, 1, "Number of loops to run");

DEFINE_int32(image_size, 3 * 64 * 64, "Size of the input image");
DEFINE_int32(joint_size, 8, "Size of the input joint");
DEFINE_int32(state_size, 50, "Size of the input state");
DEFINE_int32(pts_size, 2 * 5, "Size of the output points");

DEFINE_bool(copy_memory_once, false, "Copy memory one time");
DEFINE_string(timer_file, "", "Write timer to file");

int main(int argc, char **argv)
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << "Engine file: " << FLAGS_engine << std::endl;
    std::cout << "Total networks: " << FLAGS_total_networks << std::endl;
    std::cout << "Loops: " << FLAGS_loops << std::endl;

    std::cout << "Image size: " << FLAGS_image_size << std::endl;
    std::cout << "Joint size: " << FLAGS_joint_size << std::endl;
    std::cout << "State size: " << FLAGS_state_size << std::endl;
    std::cout << "Points size: " << FLAGS_pts_size << std::endl;

    std::cout << "Copy memory once: " << FLAGS_copy_memory_once << std::endl;
    std::cout << "Timer file: " << FLAGS_timer_file << std::endl;

    const int total = FLAGS_total_networks;

    const int image_size = FLAGS_image_size;
    const int joint_size = FLAGS_joint_size;
    const int state_size = FLAGS_state_size;
    const int pts_size = FLAGS_pts_size;

    const int all_image_size = image_size * total;
    const int all_joint_size = joint_size * total;
    const int all_state_size = state_size * total;
    const int all_pts_size = pts_size * total;

    Timer timer;
    double prepare_time, inference_time, copy_time;

    std::vector<std::unique_ptr<Network>> networks(total);
    for (int i = 0; i < total; i++)
    {
        std::string network_name = "network_" + std::to_string(i);
        networks[i] = std::make_unique<Network>(FLAGS_engine, image_size, joint_size, state_size, pts_size, network_name);
    }

    for (auto &&network : networks)
    {
        std::cout << "Network name: " << network->name() << std::endl;
    }

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // random data for testing
    std::vector<float> image(image_size);
    std::vector<float> joint(joint_size);
    std::vector<float> state_h(all_state_size);
    std::vector<float> state_c(all_state_size);
    std::vector<float> output_image(all_image_size);
    std::vector<float> output_joint(all_joint_size);
    std::vector<float> output_state_h(all_state_size);
    std::vector<float> output_state_c(all_state_size);
    std::vector<float> output_ect_pts(all_pts_size);
    std::vector<float> output_dec_pts(all_pts_size);

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
    CUDA_CHECK(cudaMalloc(&d_image, image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_joint, joint_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_state_h, all_state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_state_c, all_state_size * sizeof(float)));
    float *d_output_image, *d_output_joint, *d_output_state_h, *d_output_state_c, *d_output_ect_pts, *d_output_dec_pts;
    CUDA_CHECK(cudaMalloc(&d_output_image, all_image_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_joint, all_joint_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_state_h, all_state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_state_c, all_state_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_ect_pts, all_pts_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output_dec_pts, all_pts_size * sizeof(float)));

    // allocate bindings
    for (int i = 0; i < total; i++)
    {
        networks[i]->allocate_bindings(
            d_image,
            d_joint,
            d_state_h + i * state_size,
            d_state_c + i * state_size,
            d_output_image + i * image_size,
            d_output_joint + i * joint_size,
            d_output_state_h + i * state_size,
            d_output_state_c + i * state_size,
            d_output_ect_pts + i * pts_size,
            d_output_dec_pts + i * pts_size);
    }

    for (int l = 0; l < FLAGS_loops; l++) {
        // copy data to device
        timer.reset();
        CUDA_CHECK(cudaMemcpy(d_image, image.data(), image_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_joint, joint.data(), joint_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_state_h, state_h.data(), all_state_size * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_state_c, state_c.data(), all_state_size * sizeof(float), cudaMemcpyHostToDevice));
        prepare_time = timer.print_elapsed("Data copy to device");

        // run inference
        timer.reset();
        for (int i = 0; i < total; i++)
        {
            networks[i]->forward(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        inference_time = timer.print_elapsed("Inference");

        // copy output back to host
        timer.reset();
        if (FLAGS_copy_memory_once) {
            CUDA_CHECK(cudaMemcpy(output_image.data(), d_output_image, all_image_size * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(output_joint.data(), d_output_joint, all_joint_size * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(output_state_h.data(), d_output_state_h, all_state_size * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(output_state_c.data(), d_output_state_c, all_state_size * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(output_ect_pts.data(), d_output_ect_pts, all_pts_size * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(output_dec_pts.data(), d_output_dec_pts, all_pts_size * sizeof(float), cudaMemcpyDeviceToHost));
        } else {
            for (int i = 0; i < total; i++)
            {
                CUDA_CHECK(cudaMemcpy(output_image.data() + i * image_size, d_output_image + i * image_size, image_size * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(output_joint.data() + i * joint_size, d_output_joint + i * joint_size, joint_size * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(output_state_h.data() + i * state_size, d_output_state_h + i * state_size, state_size * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(output_state_c.data() + i * state_size, d_output_state_c + i * state_size, state_size * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(output_ect_pts.data() + i * pts_size, d_output_ect_pts + i * pts_size, pts_size * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(output_dec_pts.data() + i * pts_size, d_output_dec_pts + i * pts_size, pts_size * sizeof(float), cudaMemcpyDeviceToHost));
            }
        }
        copy_time = timer.print_elapsed("Copy output");

        if (FLAGS_timer_file != "")
        {
            std::ofstream timer_file(FLAGS_timer_file, std::ios::app);
            timer_file << prepare_time << "," << inference_time << "," << copy_time << std::endl;
        }
    }

    return 0;
}
