#include "../include/Dataset.h"
#include <fstream>
#include <iterator>
#include <stdexcept>

using namespace autoresearch;
using namespace torch;

Dataset::Dataset(const std::filesystem::path& binPath, const size_t seqLen, const torch::Device& device)
    : m_seqLen(seqLen)
    , m_device(device)
{
    std::ifstream file(binPath, std::ios::binary | std::ios::ate);

    if (!file)
    {
        throw std::runtime_error("Failed to open dataset file: " + binPath.string());
    }

    const std::streamsize byteSize = file.tellg();
    if (byteSize <= 0 || byteSize % sizeof(int32_t) != 0)
    {
        throw std::runtime_error("Dataset file size is invalid (must be a positive multiple of 4 bytes): " + binPath.string());
    }

    m_numTokens = static_cast<size_t>(byteSize / sizeof(int32_t));

    std::vector<int32_t> rawTokens(m_numTokens);

    file.seekg(0);
    file.read(reinterpret_cast<char*>(rawTokens.data()), byteSize);

    if (!file)
    {
        throw std::runtime_error("Failed to read dataset file: " + binPath.string());
    }

    m_tokensCpu = torch::tensor(rawTokens, torch::dtype(torch::kInt64));
}

torch::data::Example<> Dataset::get(const size_t index)
{
    const long offset = static_cast<long>(index * m_seqLen);
    const long length = static_cast<long>(m_seqLen);

    auto input = m_tokensCpu.narrow(0, offset, length);
    auto target = m_tokensCpu.narrow(0, offset + 1, length);

    if (!m_device.is_cpu())
    {
        input = input.to(m_device);
        target = target.to(m_device);
    }

    return {input, target};
}

torch::data::Example<> Dataset::getBatch(const size_t startIndex, const size_t batchSize)
{
    if (batchSize == 0)
    {
        throw std::invalid_argument("Batch size must be greater than zero.");
    }

    const size_t numSamples = size().value_or(0);
    if (numSamples == 0)
    {
        throw std::runtime_error("Dataset is empty.");
    }

    if (startIndex >= numSamples)
    {
        throw std::out_of_range("Dataset index out of range.");
    }

    const long seqLen = static_cast<long>(m_seqLen);
    const long batch = static_cast<long>(batchSize);

    torch::Tensor input;
    torch::Tensor target;

    if (startIndex + batchSize <= numSamples)
    {
        const long offset = static_cast<long>(startIndex * m_seqLen);
        const long tokenCount = batch * seqLen;

        input = m_tokensCpu.narrow(0, offset, tokenCount).view({batch, seqLen});
        target = m_tokensCpu.narrow(0, offset + 1, tokenCount).view({batch, seqLen});
    }
    else
    {
        std::vector<torch::Tensor> inputs;
        std::vector<torch::Tensor> targets;
        inputs.reserve(batchSize);
        targets.reserve(batchSize);

        for (size_t i = 0; i < batchSize; ++i)
        {
            const size_t idx = (startIndex + i) % numSamples;
            const long offset = static_cast<long>(idx * m_seqLen);

            inputs.push_back(m_tokensCpu.narrow(0, offset, seqLen));
            targets.push_back(m_tokensCpu.narrow(0, offset + 1, seqLen));
        }

        input = torch::stack(inputs, 0);
        target = torch::stack(targets, 0);
    }

    if (!m_device.is_cpu())
    {
        input = input.to(m_device);
        target = target.to(m_device);
    }

    return {input, target};
}

torch::optional<size_t> Dataset::size() const
{
    if (m_numTokens < m_seqLen + 1) { return 0; }
    return (m_numTokens - 1) / m_seqLen;
}
