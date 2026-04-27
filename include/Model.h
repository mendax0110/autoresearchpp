#pragma once

#include "Config.h"
#include <torch/torch.h>

namespace autoresearch
{
    /// @brief Single casual multi-head self-attention block. \struct CasualSelfAttentionImpl
    struct CasualSelfAttentionImpl : torch::nn::Module
    {
        /**
         * @brief Constructs attention from model config dimensions
         * @param numHeads Num of attention heads
         * @param headDim Dimensions per head
         * @param seqLen Max seq length
         */
        CasualSelfAttentionImpl(size_t numHeads, size_t headDim, size_t seqLen);

        torch::Tensor forward(const torch::Tensor& x);

        torch::nn::Linear qkv{nullptr};
        torch::nn::Linear proj{nullptr};
        torch::Tensor causalMask;

        size_t m_numHeads;
        size_t m_headDim;
        size_t m_embdDim;
    };
    TORCH_MODULE(CasualSelfAttention);

    /// @brief Two-layer MLP with GELU activation. \struct MlpImpl
    struct MlpImpl : torch::nn::Module
    {
        /**
         * @brief Constructs MLP from model config dimensions
         * @param embdDim Embedding dimension
         */
        explicit MlpImpl(size_t embdDim);

        torch::Tensor forward(const torch::Tensor& x);

        torch::nn::Linear fc1{nullptr};
        torch::nn::Linear fc2{nullptr};
    };
    TORCH_MODULE(Mlp);

    /// @brief Pre-norm transformer block: Layernorm -> Attnention -> Layernorm -> MLP. \struct BlockImpl
    struct BlockImpl : torch::nn::Module
    {
        /**
         * @brief Constructs a transformer block.
         * @param numHeads Num of attention heads
         * @param headDim Dimensions per head
         * @param seqLen Max seq length
         */
        BlockImpl(size_t numHeads, size_t headDim, size_t seqLen);

        torch::Tensor forward(torch::Tensor x);

        torch::nn::LayerNorm ln1{nullptr};
        torch::nn::LayerNorm ln2{nullptr};
        CasualSelfAttention attn{nullptr};
        Mlp mlp{nullptr};
    };
    TORCH_MODULE(Block);

    /// @brief Decoder-only GPT built from the project config.
    struct GptImpl : torch::nn::Module
    {
        /**
         * @brief Constructs the GPT model from the given configuration.
         * @param cfg the proj configuration containing model hyperparameters
         */
        explicit GptImpl(const Config& cfg);

        /**
         * @brief Forward pass returning logits and optional cross-entropy loss.
         * @param idx Integer token ids of shape [B, T]
         * @param targets Optional integer token ids of shape [B, T] for computing cross-entropy loss.
         * @return Pair of (logits [B, T, V] loss scalar or empty tensor).
         */
        std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor &idx, const torch::optional<torch::Tensor> &targets = {});

        torch::nn::Embedding tokEmb{nullptr};
        torch::nn::Embedding posEmb{nullptr};
        torch::nn::ModuleList blocks{nullptr};
        torch::nn::LayerNorm lnF{nullptr};
        torch::nn::Linear head{nullptr};
        torch::Tensor posIdx;

        size_t m_seqLen;
    };
    TORCH_MODULE(Gpt);
}
