#pragma once

#include "Config.h"
#include "Dataset.h"
#include "Model.h"
#include <torch/torch.h>

namespace autoresearch
{
    /// @brief Computes validation bits-per-token (BPT) on a held-out stream. \class Evaluator
    class Evaluator
    {
    public:
        /**
         * @brief Constructs an evaluator.
         * @param cfg Project config.
         * @param valDataset Validation dataset.
         * @param device Device to run evaluation on.
         */
        Evaluator(const Config& cfg, Dataset& valDataset, const torch::Device& device);

        /**
         * @brief Evaluates the model and returns val_bpb.
         *
         * Lower is better. Metric is vocab-size-independent
         *
         * @param model The model to evaluate.
         * @return Validation bits-per-byte
         */
        [[nodiscard]] double evaluate(Gpt& model) const;

    private:
        const Config& m_cfg;
        Dataset& m_dataset;
        const torch::Device& m_device;
    };
}