#pragma once

#include <string>
#include <torch/torch.h>

namespace autoresearch
{
    /// @brief Resolves and owns the torch device for the entire process lifetime. \class Device
    class DeviceWrapper
    {
    public:
        /**
         * @brief Constructs a Device from a string identifier.
         * @param deviceStr One of "cpu", "cuda" or "mps" ..
         * @throws std::runtime_error if the requested backend is not available on the system.
         */
        explicit DeviceWrapper(const std::string& deviceStr);

        /// @brief Returns the torch::Device instance.
        [[nodiscard]] const torch::Device& get() const noexcept;

        /// @brief Returns a human-readable description of the device.
        [[nodiscard]] std::string describe() const;

        /// @brief Checks if the device is a CUDA device.
        [[nodiscard]] bool isCuda() const noexcept;

        /// @brief Checks if the device is an MPS device.
        [[nodiscard]] bool isMps() const noexcept;

        /// @brief Checks if the device is a CPU device.
        [[nodiscard]] bool isCpu() const noexcept;

    private:
        torch::Device m_device;

        /**
         * @brief Resolves a string identifier to a torch::Device, checking for availability.
         * @param deviceStr The device string to resolve, e.g., "cpu", "cuda", "mps".
         * @return A torch::Device corresponding to the requested device string.
         */
        static torch::Device resolve(const std::string& deviceStr);
    };
}