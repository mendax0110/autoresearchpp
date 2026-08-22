#include "../include/CliParser.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace autoresearch;

Config CliParser::parse(const std::span<const char* const> args)
{
    Config cfg;

    for (size_t i = 1; i < args.size(); ++i)
    {
        std::string_view arg = args[i];

        auto requireNext = [&]() -> std::string_view
        {
            if (i + 1 >= args.size())
            {
                throw std::invalid_argument("Expected value after " + std::string(arg));
            }
            return args[++i];
        };

        if (arg == "--device")
        {
            cfg.device = std::string(requireNext());
        }
        else if (arg == "--gpu-id")
        {
            cfg.gpuId = std::stoi(std::string(requireNext()));
        }
        else if (arg == "--depth")
        {
            cfg.depth = std::stoull(std::string(requireNext()));
        }
        else if (arg == "--num-heads")
        {
            cfg.numHeads = std::stoull(std::string(requireNext()));
        }
        else if (arg == "--vocab-size")
        {
            cfg.vocabSize = std::stoull(std::string(requireNext()));
        }
        else if (arg == "--seq-len")
        {
            cfg.maxSeqLen = std::stoull(std::string(requireNext()));
        }
        else if (arg == "--device-batch-size")
        {
            cfg.deviceBatchSize = std::stoull(std::string(requireNext()));
        }
        else if (arg == "--total-batch-size")
        {
            cfg.totalBatchSize = std::stoull(std::string(requireNext()));
        }
        else if (arg == "--lr")
        {
            cfg.learningRate = std::stof(std::string(requireNext()));
        }
        else if (arg == "--weight-decay")
        {
            cfg.weightDecay = std::stof(std::string(requireNext()));
        }
        else if (arg == "--budget-secs")
        {
            cfg.trainBudgetSecs = std::stoull(std::string(requireNext()));
        }
        else if (arg == "--eval-tokens")
        {
            cfg.evalToken = std::stoull(std::string(requireNext()));
        }
        else if (arg == "--eval-interval-secs")
        {
            cfg.evalIntervalSecs = std::stoull(std::string(requireNext()));
        }
        else if (arg == "--data-dir")
        {
            cfg.dataDir = std::string(requireNext());
        }
        else if (arg == "--checkpoint-dir")
        {
            cfg.checkpointDir = std::string(requireNext());
        }
        else if (arg == "--log-file")
        {
            cfg.logFile = std::string(requireNext());
        }
        else if (arg == "--help" || arg == "-h")
        {
            printUsage(args[0]);
            std::exit(0);
        }
        else
        {
            throw std::invalid_argument("Unknown argument: " + std::string(arg));
        }
    }

    return cfg;
}

void CliParser::printUsage(const std::string_view programName)
{
    std::cout <<
            "Usage: " << programName << " [options]\n\n"
            "Device:\n"
            "  --device <device>           Device to use (cpu, mps, cuda, cuda:<index>)\n"
            "  --gpu-id <id>              (deprecated) GPU index to use (overridden by --device)\n\n"
            "Model:\n"
            "  --depth <n>                Number of transformer blocks (default: 8)\n"
            "  --num-heads <n>            Number of attention heads (default: 8)\n"
            "  --vocab-size <n>           Vocabulary size (default: 8192)\n"
            "  --seq-len <n>              Maximum sequence length (default: 1024)\n\n"
            "Training:\n"
            "  --device-batch-size <n>    Batch size per device (default: 64)\n"
            "  --total-batch-size <n>     Total batch size across all devices (default: 131072)\n"
            "  --lr <rate>                Learning rate (default: 0.0003)\n"
            "  --weight-decay <decay>          Weight decay factor (default: 0.1)\n"
            "  --budget-secs <seconds>    Training time budget in seconds (default: 300)\n\n"
            "Evaluation:\n"
            "  --eval-tokens <n>         Number of tokens to evaluate on (default: 1048576)\n"
            "  --eval-interval-secs <n>  Evaluation interval in seconds (default: 60)\n\n"
            "I/O:\n"
            "  --data-dir <path>         Directory containing dataset files (default: \"data\")\n"
            "  --checkpoint-dir <path>   Directory to save checkpoints (default: \"checkpoints\")\n"
            "  --log-file <path>        Path to experiment log file (default: \"experiments.jsonl\")\n\n"
            "Other:\n"
            "  --help, -h                Show this help message and exit\n";
}
