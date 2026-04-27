#include "../include/Config.h"
#include <gtest/gtest.h>

using namespace autoresearch;

TEST(ConfigTest, DefaultValues)
{
    Config cfg;

    EXPECT_EQ(cfg.totalBatchSize, 1u << 17);
    EXPECT_EQ(cfg.deviceBatchSize, 64);
    EXPECT_EQ(cfg.maxSeqLen, 1024);
    EXPECT_EQ(cfg.trainBudgetSecs, 300);

    EXPECT_EQ(cfg.vocabSize, 8192);
    EXPECT_EQ(cfg.depth, 8);
    EXPECT_EQ(cfg.headDim, 64);
    EXPECT_EQ(cfg.numHeads, 8);

    EXPECT_FLOAT_EQ(cfg.learningRate, 3e-4f);
    EXPECT_FLOAT_EQ(cfg.weightDecay, 0.1f);
    EXPECT_FLOAT_EQ(cfg.beta1, 0.9f);
    EXPECT_FLOAT_EQ(cfg.beta2, 0.95f);

    EXPECT_EQ(cfg.evalToken, 1u << 20);
    EXPECT_EQ(cfg.evalIntervalSecs, 60);

    EXPECT_EQ(cfg.dataDir, "data");
    EXPECT_EQ(cfg.checkpointDir, "checkpoints");
    EXPECT_EQ(cfg.logFile, "experiments.jsonl");
    EXPECT_EQ(cfg.metricsFile, "metrics.jsonl");

    EXPECT_EQ(cfg.device, "cpu");
    EXPECT_EQ(cfg.gpuId, 0);
}

TEST(ConfigTest, TotalBatchSizeDefault)
{
    const Config cfg;
    const auto n = cfg.totalBatchSize;
    EXPECT_GT(n, 0u);
    EXPECT_EQ(n & (n - 1), 0u) << "totalBatchSize should be a power of 2 for optimal performance.";
}