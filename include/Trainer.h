#pragma once

#include "Config.h"
#include "Dataset.h"
#include "Evaluator.h"
#include "Model.h"
#include "Metricslog.h"
#include <functional>
#include <memory>
#include <torch/torch.h>

namespace autoresearch
{
    /// @brief Callback invoked after each eval, receives step and val_bpb. \using EvalCallback
    using EvalCallback = std::function<void(size_t step, double valBpb)>;

    /// @brief Runs the timed training loop against a fixed wall-clock budget. \class Trainer
    class Trainer
    {
    public:
        /**
         * @brief Constructs a trainer.
         * @param cfg Project config.
         * @param model GPT mpdel (moved into trainer ownership)
         * @param trainSet trainer dataset
         * @param evaluator Evaluator for periodic val_bpb meas
         * @param device Device to run training on
         */
        Trainer(const Config& cfg, Gpt model, Dataset& trainSet, std::unique_ptr<Evaluator> evaluator, MetricsLog& metricsLog, const torch::Device& device);

        /**
         * @brief Runs training for the configured wall-clock budget.
         * @param onEval Optional callback invoked after each eval pass.
         * @return Final val_bpb achieved before the budget expired.
         */
        double run(const EvalCallback& onEval = nullptr);

        /// @brief Returns a reference to the trainer's model.
        [[nodiscard]] Gpt& model() noexcept;

    private:
        const Config& m_cfg;
        Gpt m_model;
        Dataset& m_trainSet;
        std::unique_ptr<Evaluator> m_evaluator;
        MetricsLog& m_metricsLog;
        const torch::Device& m_device;
        std::unique_ptr<torch::optim::AdamW> m_optimizer;

        /**
         * @brief Scales the gradients of all model params by the given factor.
         * @param scale The factor to scale gradients by.
         */
        void scaleGradients(float scale);
    };
}