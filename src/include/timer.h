#pragma once

#include <chrono>
#include <iostream>

class Timer
{
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset()
    {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed() const
    {
        return std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now() - start_).count();
    }

    double print_elapsed(const std::string &message)
    {
        double elapsed_time = elapsed() * 1000;
        std::cout << message << ": " << elapsed_time << "ms" << std::endl;
        return elapsed_time;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};
