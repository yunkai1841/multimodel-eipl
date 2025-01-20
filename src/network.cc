#include "network.h"

Network::Network(std::string engine_path,
                 int image_size,
                 int joint_size,
                 int state_size,
                 int pts_size,
                 std::string name)
    : image_size_(image_size),
      joint_size_(joint_size),
      state_size_(state_size),
      pts_size_(pts_size),
      name_(name)
{
    init_engine(engine_path);
}

void Network::forward(cudaStream_t stream)
{
    // Execute the network
    if (!context_->enqueueV3(stream))
    {
        std::cerr << "Error: could not run inference" << std::endl;
        exit(1);
    }
}

// Load the TensorRT engine from the file and create the execution context
void Network::init_engine(std::string engine_path)
{
    Logger logger;
    auto runtime = nvinfer1::createInferRuntime(logger);
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file)
    {
        std::cerr << "Error: could not open engine file" << std::endl;
        exit(1);
    }

    engine_file.seekg(0, engine_file.end);
    size_t engine_size = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);

    std::vector<char> engine_data(engine_size);
    engine_file.read(engine_data.data(), engine_size);
    engine_file.close();

    auto engine = runtime->deserializeCudaEngine(engine_data.data(), engine_size);
    if (!engine)
    {
        std::cerr << "Error: could not deserialize engine" << std::endl;
        exit(1);
    }

    // Create the execution context
    context_ = engine->createExecutionContext();
    if (!context_)
    {
        std::cerr << "Error: could not create execution context" << std::endl;
        exit(1);
    }
}

// Allocate bindings for engine inputs
void Network::allocate_bindings(float *input_image, float *input_joint, float *input_state_h, float *input_state_c,
                                float *output_image, float *output_joint, float *output_state_h, float *output_state_c,
                                float *output_ect_pts, float *output_dec_pts)
{
    set_binding("i.image", input_image);
    set_binding("i.joint", input_joint);
    set_binding("i.state_h", input_state_h);
    set_binding("i.state_c", input_state_c);
    set_binding("o.image", output_image);
    set_binding("o.joint", output_joint);
    set_binding("o.state_h", output_state_h);
    set_binding("o.state_c", output_state_c);
    set_binding("o.enc_pts", output_ect_pts);
    set_binding("o.dec_pts", output_dec_pts);
}

void Network::set_binding(const std::string &name, void *ptr)
{
    if (!context_->setTensorAddress(name.c_str(), ptr))
    {
        std::cerr << "Error: could not set tensor address" << std::endl;
        exit(1);
    }
}
