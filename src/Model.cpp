#include "../include/Model.h"
#include <cmath>
#include <stdexcept>

using namespace autoresearch;
using namespace torch;

CasualSelfAttentionImpl::CasualSelfAttentionImpl(const size_t numHeads, const size_t headDim, const size_t seqLen)
    : m_numHeads(numHeads)
    , m_headDim(headDim)
    , m_embdDim(numHeads * headDim)
{
    qkv = register_module("qkv", nn::Linear(nn::LinearOptions(m_embdDim, 3 * m_embdDim).bias(false)));
    proj = register_module("proj",nn::Linear(nn::LinearOptions(m_embdDim, m_embdDim).bias(false)));

    const auto mask = torch::ones({static_cast<long>(seqLen), static_cast<long>(seqLen)}, torch::dtype(torch::kBool)).triu(1);
    causalMask = register_buffer("causalMask", mask);
}

torch::Tensor CasualSelfAttentionImpl::forward(const torch::Tensor& x)
{
    const auto [B, T, C] = std::tuple{x.size(0), x.size(1), x.size(2)};

    const auto qkv_out = qkv->forward(x).reshape({B, T, 3, static_cast<long>(m_numHeads), static_cast<long>(m_headDim)}).permute({2, 0, 3, 1, 4});

    const auto q = qkv_out[0];
    const auto k = qkv_out[1];
    const auto v = qkv_out[2];

    const float scale = 1.0f / std::sqrt(static_cast<float>(m_headDim));

    auto attn = torch::matmul(q, k.transpose(-2, -1)) * scale;

    // causal mask, upper-triangular filled with -inf.
    const auto mask = causalMask.narrow(0, 0, T).narrow(1, 0, T);
    attn.masked_fill_(mask.unsqueeze(0).unsqueeze(0), -std::numeric_limits<float>::infinity());

    attn = torch::softmax(attn, -1);

    const auto out = torch::matmul(attn, v).transpose(1, 2).contiguous().reshape({B, T, C});

    return proj->forward(out);
}

MlpImpl::MlpImpl(size_t embdDim)
{
    fc1 = register_module("fc1", nn::Linear(embdDim, 4 * embdDim));
    fc2 = register_module("fc2", nn::Linear(4 * embdDim, embdDim));
}

torch::Tensor MlpImpl::forward(const torch::Tensor& x)
{
    return fc2->forward(torch::gelu(fc1->forward(x), "tanh"));
}

BlockImpl::BlockImpl(size_t numHeads, size_t headDim, size_t seqLen)
{
    const size_t embedDim = numHeads * headDim;

    ln1 = register_module("ln1", nn::LayerNorm(nn::LayerNormOptions({static_cast<long>(embedDim)})));
    ln2 = register_module("ln2", nn::LayerNorm(nn::LayerNormOptions({static_cast<long>(embedDim)})));
    attn = register_module("attn", CasualSelfAttention(numHeads, headDim, seqLen));
    mlp = register_module("mlp", Mlp(embedDim));
}

torch::Tensor BlockImpl::forward(torch::Tensor x)
{
    x = x + attn->forward(ln1->forward(x));
    x = x + mlp->forward(ln2->forward(x));
    return x;
}

GptImpl::GptImpl(const Config &cfg) : m_seqLen(cfg.maxSeqLen)
{
    const size_t embedDim = cfg.numHeads * cfg.headDim;

    tokEmb = register_module("tokEmb", nn::Embedding(cfg.vocabSize, embedDim));
    posEmb = register_module("posEmb", nn::Embedding(cfg.maxSeqLen, embedDim));

    blocks = register_module("blocks", nn::ModuleList());
    for (size_t i = 0; i < cfg.depth; ++i)
    {
        blocks->push_back(Block(cfg.numHeads, cfg.headDim, cfg.maxSeqLen));
    }

    lnF = register_module("lnF", nn::LayerNorm(nn::LayerNormOptions({static_cast<long>(embedDim)})));
    head = register_module("head", nn::Linear(nn::LinearOptions(static_cast<long>(embedDim), cfg.vocabSize).bias(false)));

    // weight-decay: token embeddin shares weights with the output projection
    //head->weight = tokEmb->weight;

    posIdx = register_buffer("posIdx", torch::arange(static_cast<long>(cfg.maxSeqLen), torch::dtype(torch::kLong)));
}

std::pair<torch::Tensor, torch::Tensor> GptImpl::forward(const torch::Tensor& idx, const torch::optional<torch::Tensor>& targets)
{
    const long T = idx.size(1);

    if (T > static_cast<long>(m_seqLen))
    {
        throw std::invalid_argument("Input sequence length " + std::to_string(T) + " exceeds model's max sequence length of " + std::to_string(m_seqLen));
    }

    const auto pos = posIdx.narrow(0, 0, T);
    auto x = tokEmb->forward(idx) + posEmb->forward(pos);

    for (const auto& block : *blocks)
    {
        x = block->as<BlockImpl>()->forward(x);
    }

    x = lnF->forward(x);
    auto logits = head->forward(x);

    torch::Tensor loss;
    if (targets.has_value())
    {
        const long B = logits.size(0);
        const long V = logits.size(2);

        loss = torch::cross_entropy_loss(logits.view({B * T, V}), targets->view({B * T}));
    }

    return {logits, loss};
}
