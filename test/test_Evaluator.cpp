#include "../include/Config.h"
#include "../include/Dataset.h"
#include "../include/Evaluator.h"
#include "../include/Model.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <vector>

using namespace autoresearch;

namespace
{
    std::filesystem::path makeTempDataPath()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() / ("evaluator-test-" + std::to_string(stamp) + ".bin");
    }

    void writeInt32Bin(const std::filesystem::path& path, const std::vector<int32_t>& tokens)
    {
        std::ofstream out(path, std::ios::binary);
        ASSERT_TRUE(out.is_open());
        out.write(reinterpret_cast<const char*>(tokens.data()), static_cast<std::streamsize>(tokens.size() * sizeof(int32_t)));
        ASSERT_TRUE(out.good());
    }
}

TEST(EvaluatorTest, ReturnsFiniteBpbAndRestoresTrainMode)
{
    Config cfg;
    cfg.vocabSize = 64;
    cfg.depth = 1;
    cfg.numHeads = 2;
    cfg.headDim = 8;
    cfg.maxSeqLen = 4;
    cfg.evalToken = 16;
    cfg.deviceBatchSize = 2;

    const auto path = makeTempDataPath();
    std::vector<int32_t> tokens;
    tokens.reserve(32);
    for (int32_t i = 0; i < 32; ++i)
    {
        tokens.push_back(i % static_cast<int32_t>(cfg.vocabSize));
    }
    writeInt32Bin(path, tokens);

    const torch::Device device(torch::kCPU);
    Dataset valSet(path, cfg.maxSeqLen, device);
    Gpt model(cfg);
    model->train();

    const Evaluator evaluator(cfg, valSet, device);
    const double bpb = evaluator.evaluate(model);

    EXPECT_TRUE(std::isfinite(bpb));
    EXPECT_GT(bpb, 0.0);
    EXPECT_TRUE(model->is_training());

    std::filesystem::remove(path);
}
