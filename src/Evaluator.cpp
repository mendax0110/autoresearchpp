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
            auto example = m_dataset.get(i + j);
            inputs.push_back(example.data);
            targets.push_back(example.target);
        }

        auto inputBatch = torch::stack(inputs, 0);
        auto targetBatch = torch::stack(targets, 0);

        auto [logits, loss] = model->forward(inputBatch, targetBatch);

        totalLoss += loss.item<double>() * static_cast<double>(currentBatch * m_cfg.maxSeqLen);
        totalTokens += static_cast<double>(currentBatch * m_cfg.maxSeqLen);
    }

    model->train();

    if (totalTokens == 0.0) { return std::numeric_limits<double>::infinity(); }

    // convert cross-entropy loss to bits-per-byte (bpb)
    // using log2(e) factor ; bpb = loss_nats / ln(2)
    const double avgLoss = totalLoss / totalTokens;
    const double bpb = avgLoss / std::log(2.0);
    return bpb;
}
