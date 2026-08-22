#include "../include/Device.h"
#include <stdexcept>
#include <string>

using namespace autoresearch;
using namespace torch;

DeviceWrapper::DeviceWrapper(const std::string& deviceStr) : m_device(resolve(deviceStr))
{

}

const torch::Device& DeviceWrapper::get() const noexcept
{
    return m_device;
}

std::string DeviceWrapper::describe() const
{
    switch (m_device.type())
    {
        case torch::kCUDA:
            return "CUDA (GPU " + std::to_string(m_device.index()) + ")";
        case torch::kMPS:
            return "MPS (Apple GPU)";
        case torch::kCPU:
            return "CPU";
        default:
            return "Unknown Device";
    }
}

bool DeviceWrapper::isCuda() const noexcept
{
    return m_device.is_cuda();
}

bool DeviceWrapper::isMps() const noexcept
{
    return m_device.is_mps();
}

bool DeviceWrapper::isCpu() const noexcept
{
    return m_device.is_cpu();
}

torch::Device DeviceWrapper::resolve(const std::string& deviceStr)
{
    if (deviceStr == "cpu")
    {
        return {torch::kCPU};
    }

    if (deviceStr == "mps")
    {
#ifdef __APPLE__
        if (!hasMPS())
        {
            throw std::runtime_error("MPS backend requested but not available on this system.");
        }
        return {torch::kMPS};
#else
        throw std::runtime_error("MPS backend is only supported on Apple Silicon devices.");
#endif
    }

    if (deviceStr == "cuda" || deviceStr.starts_with("cuda:"))
    {
        if (!torch::cuda::is_available())
        {
            throw std::runtime_error("CUDA backend requested but no CUDA devices are available.");
        }

        int idx = 0;
        if (deviceStr.starts_with("cuda:"))
        {
            idx = std::stoi(deviceStr.substr(5));
        }
        return {torch::kCUDA, static_cast<DeviceIndex>(idx)};
    }

    throw std::invalid_argument("Unknown device string: \"" + deviceStr + R"(". Valid options are "cpu", "mps", "cuda" or "cuda:<index>".)");
}
