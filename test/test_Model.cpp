#include "../include/Config.h"
#include "../include/Model.h"
#include <gtest/gtest.h>
#include <stdexcept>

using namespace autoresearch;

namespace
{
    Config tinyConfig()
    {
        Config cfg;
        cfg.vocabSize = 64;
        cfg.depth = 2;
        cfg.numHeads = 2;
        cfg.headDim = 8;
        cfg.maxSeqLen = 8;
        return cfg;
    }
}

TEST(ModelTest, ForwardReturnsExpectedShapes)
{
    const Config cfg = tinyConfig();
    Gpt model(cfg);

    const auto idx = torch::randint(static_cast<int64_t>(cfg.vocabSize), {2, 4}, torch::dtype(torch::kLong));
    const auto [logits, loss] = model->forward(idx, idx);

    EXPECT_EQ(logits.sizes(), torch::IntArrayRef({2, 4, static_cast<long>(cfg.vocabSize)}));
    EXPECT_TRUE(loss.defined());
    EXPECT_TRUE(torch::isfinite(loss).item<bool>());
}

TEST(ModelTest, ForwardWithoutTargetsReturnsUndefinedLoss)
{
    const Config cfg = tinyConfig();
    Gpt model(cfg);

    const auto idx = torch::randint(static_cast<int64_t>(cfg.vocabSize), {1, 3}, torch::dtype(torch::kLong));
    const auto [logits, loss] = model->forward(idx);

    EXPECT_EQ(logits.sizes(), torch::IntArrayRef({1, 3, static_cast<long>(cfg.vocabSize)}));
    EXPECT_FALSE(loss.defined());
}

TEST(ModelTest, ThrowsWhenSequenceTooLong)
{
    const Config cfg = tinyConfig();
    Gpt model(cfg);

    const auto idx = torch::randint(static_cast<int64_t>(cfg.vocabSize), {1, static_cast<long>(cfg.maxSeqLen + 1)}, torch::dtype(torch::kLong));
    EXPECT_THROW(static_cast<void>(model->forward(idx)), std::invalid_argument);
}

TEST(ModelTest, CachedBuffersAreInitialized)
{
    const Config cfg = tinyConfig();
    Gpt model(cfg);

    EXPECT_TRUE(model->posIdx.defined());
    EXPECT_EQ(model->posIdx.sizes(), torch::IntArrayRef({static_cast<long>(cfg.maxSeqLen)}));

    CasualSelfAttention attn(cfg.numHeads, cfg.headDim, cfg.maxSeqLen);
    EXPECT_TRUE(attn->causalMask.defined());
    EXPECT_EQ(attn->causalMask.sizes(), torch::IntArrayRef({static_cast<long>(cfg.maxSeqLen), static_cast<long>(cfg.maxSeqLen)}));
}
