#pragma once

#include "Config.h"
#include "Model.h"
#include <memory>
#include <torch/torch.h>

namespace autoresearch
{
    /// @brief Wraps AdamW with weight-decay param group separation. \class OptimizerFactory
    class OptimizerFactory
    {
    public:
        /**
         * @brief Creates an AdamW optimizer with two param groups.
         *
         * Parameters with ndim >= 2 receive weight decay; bioases and
         * layerNorm weights are excluded form decay.
         *
         * @param model The model whose params are to be optimized
         * @param cfg Training config (lr, betas, weight_decay)
         * @return Heap-allocated AdamW optimizer
         */
        [[nodiscard]] static std::unique_ptr<torch::optim::AdamW> makeAdamW(const Gpt& model, const Config& cfg);
    };
}