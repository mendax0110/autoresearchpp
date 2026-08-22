#include "../include/Dataset.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

using namespace autoresearch;

namespace
{
    std::filesystem::path makeTempPath(const std::string& prefix)
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() / (prefix + std::to_string(stamp) + ".bin");
    }

    void writeInt32Bin(const std::filesystem::path& path, const std::vector<int32_t>& tokens)
    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open());
        out.write(reinterpret_cast<const char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(int32_t)));
        ASSERT_TRUE(out.good());
    }
}

TEST(DatasetTest, SizeAndGetOnCpu)
{
    const auto path = makeTempPath("dataset-test-");
    writeInt32Bin(path, {1, 2, 3, 4, 5, 6, 7});

    const torch::Device device(torch::kCPU);
    Dataset ds(path, 3, device);

    ASSERT_TRUE(ds.size().has_value());
    EXPECT_EQ(*ds.size(), 2u);

    const auto ex0 = ds.get(0);
    EXPECT_TRUE(ex0.data.device().is_cpu());
    EXPECT_TRUE(ex0.target.device().is_cpu());
    EXPECT_EQ(ex0.data.dtype(), torch::kInt64);
    EXPECT_EQ(ex0.target.dtype(), torch::kInt64);
    EXPECT_EQ(ex0.data.sizes(), torch::IntArrayRef({3}));
    EXPECT_EQ(ex0.target.sizes(), torch::IntArrayRef({3}));
    EXPECT_EQ(ex0.data[0].item<int64_t>(), 1);
    EXPECT_EQ(ex0.data[1].item<int64_t>(), 2);
    EXPECT_EQ(ex0.data[2].item<int64_t>(), 3);
    EXPECT_EQ(ex0.target[0].item<int64_t>(), 2);
    EXPECT_EQ(ex0.target[1].item<int64_t>(), 3);
    EXPECT_EQ(ex0.target[2].item<int64_t>(), 4);

    const auto ex1 = ds.get(1);
    EXPECT_EQ(ex1.data[0].item<int64_t>(), 4);
    EXPECT_EQ(ex1.data[1].item<int64_t>(), 5);
    EXPECT_EQ(ex1.data[2].item<int64_t>(), 6);
    EXPECT_EQ(ex1.target[0].item<int64_t>(), 5);
    EXPECT_EQ(ex1.target[1].item<int64_t>(), 6);
    EXPECT_EQ(ex1.target[2].item<int64_t>(), 7);

    std::filesystem::remove(path);
}

TEST(DatasetTest, GetBatchReturnsExpectedSlicesAndWraps)
{
    const auto path = makeTempPath("dataset-batch-");
    writeInt32Bin(path, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});

    const torch::Device device(torch::kCPU);
    Dataset ds(path, 3, device);

    // samples: [1,2,3]->[2,3,4], [4,5,6]->[5,6,7], [7,8,9]->[8,9,10]
    auto batch = ds.getBatch(0, 2);
    EXPECT_EQ(batch.data.sizes(), torch::IntArrayRef({2, 3}));
    EXPECT_EQ(batch.target.sizes(), torch::IntArrayRef({2, 3}));
    EXPECT_EQ(batch.data[0][0].item<int64_t>(), 1);
    EXPECT_EQ(batch.data[1][0].item<int64_t>(), 4);
    EXPECT_EQ(batch.target[1][2].item<int64_t>(), 7);

    // wrap at dataset end: indices 2 then 0
    auto wrapped = ds.getBatch(2, 2);
    EXPECT_EQ(wrapped.data[0][0].item<int64_t>(), 7);
    EXPECT_EQ(wrapped.data[1][0].item<int64_t>(), 1);

    EXPECT_THROW(static_cast<void>(ds.getBatch(0, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(ds.getBatch(999, 1)), std::out_of_range);

    std::filesystem::remove(path);
}

TEST(DatasetTest, InvalidByteSizeThrows)
{
    const auto path = makeTempPath("dataset-bad-");
    std::ofstream out(path, std::ios::binary);
    constexpr char bytes[3] = {'a', 'b', 'c'};
    out.write(bytes, sizeof(bytes));
    out.close();

    const torch::Device device(torch::kCPU);
    EXPECT_THROW(static_cast<void>(Dataset(path, 4, device)), std::runtime_error);

    std::filesystem::remove(path);
}
