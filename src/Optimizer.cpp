#include "../include/Optimizer.h"
#include <string>
#include <vector>

using namespace autoresearch;
using namespace torch;

std::unique_ptr<torch::optim::AdamW> OptimizerFactory::makeAdamW(const Gpt& model, const Config& cfg)
{
    std::vector<torch::Tensor> decayParams;
    std::vector<torch::Tensor> noDecayParams;

    for (const auto& item : model->named_parameters())
    {
        const auto& param = item.value();

        if (!param.requires_grad()) { continue; }

        if (param.dim() < 2)
        {
            noDecayParams.push_back(param);
        }
        else
        {
            decayParams.push_back(param);
        }
    }

    std::vector<torch::optim::AdamWOptions> groupOptions;

    torch::optim::AdamWOptions decayOpts(cfg.learningRate);
    decayOpts.betas({cfg.beta1, cfg.beta2});
    decayOpts.weight_decay(cfg.weightDecay);

    torch::optim::AdamWOptions noDecayOpts(cfg.learningRate);
    noDecayOpts.betas({cfg.beta1, cfg.beta2});
    noDecayOpts.weight_decay(0.0);

    std::vector<torch::optim::OptimizerParamGroup> groups;
    groups.emplace_back(decayParams, std::make_unique<torch::optim::AdamWOptions>(decayOpts));
    groups.emplace_back(noDecayParams, std::make_unique<torch::optim::AdamWOptions>(noDecayOpts));

    return std::make_unique<torch::optim::AdamW>(groups, decayOpts);
}
