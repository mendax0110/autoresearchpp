#include "../include/Device.h"
#include <gtest/gtest.h>
#include <stdexcept>

using namespace autoresearch;

TEST(DeviceTest, CpuDeviceResolves)
{
    DeviceWrapper device("cpu");
    EXPECT_TRUE(device.isCpu());
    EXPECT_FALSE(device.isCuda());
    EXPECT_FALSE(device.isMps());
    EXPECT_EQ(device.describe(), "CPU");
}

TEST(DeviceTest, UnknownDeviceThrows)
{
    EXPECT_THROW(static_cast<void>(DeviceWrapper("banana")), std::invalid_argument);
}

TEST(DeviceTest, CudaResolutionMatchesAvailability)
{
    if (torch::cuda::is_available())
    {
        DeviceWrapper device("cuda");
        EXPECT_TRUE(device.isCuda());
    }
    else
    {
        EXPECT_THROW(static_cast<void>(DeviceWrapper("cuda")), std::runtime_error);
    }
}
