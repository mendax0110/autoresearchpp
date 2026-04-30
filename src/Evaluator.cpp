#include "../include/Evaluator.h"
#include <algorithm>
#include <cmath>
#include <vector>

using namespace autoresearch;
using namespace torch;

Evaluator::Evaluator(const Config& cfg, Dataset& valDataset, const torch::Device& device)
    : m_cfg(cfg)
    , m_dataset(valDataset)
    , m_device(device)
{

}

double Evaluator::evaluate(Gpt& model) const
{
    model->eval();
    torch::NoGradGuard noGrad;

    double totalLoss = 0.0;
    double totalTokens = 0.0;

    const size_t numSamples = m_dataset.size().value_or(0);
    const size_t evalSamples = std::min(numSamples, m_cfg.evalToken / m_cfg.maxSeqLen);
    const size_t batchSize = std::max<size_t>(1, m_cfg.deviceBatchSize);

    for (size_t i = 0; i < evalSamples; i += batchSize)
    {
        const size_t currentBatch = std::min(batchSize, evalSamples - i);

        std::vector<torch::Tensor> inputs;
        std::vector<torch::Tensor> targets;
        inputs.reserve(currentBatch);
        targets.reserve(currentBatch);

        for (size_t j = 0; j < currentBatch; ++j)
        {
            auto [input, target] = m_dataset.get((i + j) % numSamples);
            inputs.push_back(input.to(m_device));
            targets.push_back(target.to(m_device));
        }

        auto inputBatch = torch::stack(inputs);
        auto targetBatch = torch::stack(targets);

        auto output = model->forward(inputBatch);
        auto loss = torch::nn::functional::cross_entropy(output, targetBatch, {}, torch::Reduction::Sum);

        totalLoss += loss.item<double>();
        totalTokens += targetBatch.numel();
    }

    // Avoid division by zero in case of empty dataset
    return totalTokens > 0 ? totalLoss / (totalTokens * std::log(2)) : std::numeric_limits<double>::infinity();
}
