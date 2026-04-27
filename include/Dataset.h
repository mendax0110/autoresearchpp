#pragma once

#include "Config.h"
#include <filesystem>
#include <memory>
#include <span>
#include <vector>
#include <torch/torch.h>

namespace autoresearch
{
    class Dataset : public torch::data::Dataset<Dataset>
    {
    public:
        /**
         * @brief Opens a Binary token file
         * @param binPath path to the .bin file produced by prepare
         * @param seqLen Sequence length for training samples
         * @param device Device to load the dataset tensors onto
         */
        Dataset(const std::filesystem::path& binPath, size_t seqLen, const torch::Device& device);

        /// @brief Returns the (input, target) tensor pair for the given index.
        torch::data::Example<> get(size_t index) override;

        /// @brief Returns a batched (input, target) tensor pair for contiguous samples.
        [[nodiscard]] torch::data::Example<> getBatch(size_t startIndex, size_t batchSize);

        /// @brief Number of full sequences available.
        [[nodiscard]] torch::optional<size_t> size() const override;

    private:
        torch::Tensor m_tokensCpu;
        size_t m_numTokens = 0;
        size_t m_seqLen;
        torch::Device m_device;
    };
}