#pragma once

#include <chrono>
#include <iostream>
#include <iomanip>

#include "NvInfer.h"

class Logger : public nvinfer1::ILogger
{
public:
    Logger(Severity severity = Severity::kWARNING)
        : severity_(severity)
    {
    }

    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity > severity_)
            return;

        auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::cerr << "[" << std::put_time(std::localtime(&now), "%Y-%m-%d %H:%M:%S") << "] ";

        switch (severity)
        {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "[INTERNAL_ERROR] ";
            break;
        case Severity::kERROR:
            std::cerr << "[E] ";
            break;
        case Severity::kWARNING:
            std::cerr << "[W] ";
            break;
        case Severity::kINFO:
            std::cerr << "[I] ";
            break;
        case Severity::kVERBOSE:
            std::cerr << "[V] ";
            break;
        }

        std::cerr << msg << std::endl;
    }

private:
    Severity severity_{Severity::kWARNING};
};
