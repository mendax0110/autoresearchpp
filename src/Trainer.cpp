#include "../include/Trainer.h"
#include "../include/Optimizer.h"
#include <chrono>
#include <cstdio>
#include <stdexcept>

using namespace autoresearch;
using namespace torch;

Trainer::Trainer(const Config& cfg, Gpt model, Dataset& trainSet, std::unique_ptr<Evaluator> evaluator, MetricsLog& metricsLog, const torch::Device& device)
    : m_cfg(cfg)
    , m_model(std::move(model))
    , m_trainSet(trainSet)
    , m_evaluator(std::move(evaluator))
    , m_metricsLog(metricsLog)
    , m_device(device)
    , m_optimizer(OptimizerFactory::makeAdamW(m_model, cfg))
{
    m_model->to(m_device);
    m_model->train();
}

double Trainer::run(EvalCallback onEval)
{
    const size_t gradAccumSteps = m_cfg.totalBatchSize / (m_cfg.deviceBatchSize * m_cfg.maxSeqLen);

    if (gradAccumSteps == 0)
    {
        throw std::invalid_argument("Total batch size must be at least device batch size * max sequence length.");
    }

    const auto budgetDuration = std::chrono::seconds(m_cfg.trainBudgetSecs);
    const auto evalInterval = std::chrono::seconds(m_cfg.trainBudgetSecs);
    const auto startTime = std::chrono::steady_clock::now();
    auto lastEvalTime = startTime;

    double lastValBpb = 0.0;
    size_t step = 0;
    size_t dataIndex = 0;
    const size_t datasetSize = m_trainSet.size().value_or(0);

    if (datasetSize == 0)
    {
        throw std::runtime_error("Training dataset is empty.");
    }

    const size_t effectiveGradAccumSteps = (gradAccumSteps > 2) ? 2 : gradAccumSteps;

    while (std::chrono::steady_clock::now() - startTime < budgetDuration)
    {
        m_optimizer->zero_grad();

        float accumulatedLoss = 0.0f;

        for (size_t micro = 0; micro < effectiveGradAccumSteps; ++micro)
        {
            auto example = m_trainSet.get(dataIndex % datasetSize);
            ++dataIndex;

            auto input = example.data.unsqueeze(0);
            auto target = example.target.unsqueeze(0);

            auto [logits, loss] = m_model->forward(input, target);

            // scale loss for gradient accumulation
            auto scaledLoss = loss / static_cast<float>(effectiveGradAccumSteps);
            scaledLoss.backward();

            accumulatedLoss += loss.item<float>();
        }

        // gradient clipping before the optimizer step.
        torch::nn::utils::clip_grad_norm_(m_model->parameters(), 1.0f);
        m_optimizer->step();
        ++step;

        const auto now = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count() / 1000.0;
        const float loss = accumulatedLoss / static_cast<float>(effectiveGradAccumSteps);
        //m_metricsLog.append({step, elapsed, loss, 0.0});
        double valBpbThisStep = 0.0;
        if (now - lastEvalTime >= evalInterval)
        {
            valBpbThisStep = m_evaluator->evaluate(m_model);
            lastEvalTime = now;
            lastValBpb = valBpbThisStep;

            std::printf("[step %5zu | %6.1fs] val_bpb = %.4f\n", step, elapsed, lastValBpb);

            if (onEval)
            {
                onEval(step, lastValBpb);
            }
        }
        m_metricsLog.append({step, elapsed, loss, valBpbThisStep});
    }

    // final eval at end of budget
    lastValBpb = m_evaluator->evaluate(m_model);
    std::printf("[step %5zu | budget expired] final val_bpb = %.4f\n", step, lastValBpb);

    if (onEval)
    {
        onEval(step, lastValBpb);
    }

    return lastValBpb;
}

Gpt &Trainer::model() noexcept
{
    return m_model;
}
