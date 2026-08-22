#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace autoresearch
{
    /**
     * @brief Struct representing the configuration \struct Config
     */
    struct Config
    {
        // Training
        size_t totalBatchSize = 1u << 17;
        size_t deviceBatchSize = 64;
        size_t maxSeqLen = 1024;
        size_t trainBudgetSecs = 300;

        // Model
        size_t vocabSize = 8192;
        size_t depth = 8;
        size_t headDim = 64;
        size_t numHeads = 8;

        // Optimizer
        float learningRate = 3e-4f;
        float weightDecay = 0.1f;
        float beta1 = 0.9f;
        float beta2 = 0.95f;

        // Evaluation
        size_t evalToken = 1u << 20;
        size_t evalIntervalSecs = 60;

        // I/O
        std::string dataDir = "data";
        std::string checkpointDir = "checkpoints";
        std::string logFile = "experiments.jsonl";
        std::string metricsFile = "metrics.jsonl";

        // Device
        std::string device = "cpu";
        int gpuId = 0;
    };
}