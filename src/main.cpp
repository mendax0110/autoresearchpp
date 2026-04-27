#include "../include/CliParser.h"
#include "../include/Config.h"
#include "../include/Dataset.h"
#include "../include/Device.h"
#include "../include/Evaluator.h"
#include "../include/ExperimentLog.h"
#include "../include/Metricslog.h"
#include "../include/Model.h"
#include "../include/Trainer.h"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>

using namespace autoresearch;

namespace
{
    std::string currentTimestamp()
    {
        const auto now = std::chrono::system_clock::now();
        const auto timeT = std::chrono::system_clock::to_time_t(now);
        std::ostringstream oss;
        oss << std::put_time(std::gmtime(&timeT), "%Y-%m-%dT%H:%M:%SZ");
        return oss.str();
    }
}

int main(int argc, char* argv[])
{
    Config cfg;

    try
    {
        cfg = CliParser::parse({argv, static_cast<size_t>(argc)});
    }
    catch (const std::invalid_argument& e)
    {
        std::cerr << "Error parsing arguments: " << e.what() << std::endl;
        CliParser::printUsage(argv[0]);
        return EXIT_FAILURE;
    }

    try
    {
        DeviceWrapper device(cfg.device);
        std::printf("Using device: %s\n", device.describe().c_str());

        const std::filesystem::path dataDir       = cfg.dataDir;
        const std::filesystem::path trainPath     = dataDir / "train.bin";
        const std::filesystem::path valPath       = dataDir / "val.bin";
        const std::filesystem::path checkpointDir = cfg.checkpointDir;
        std::filesystem::create_directories(checkpointDir);

        std::printf("Loading datasets from %s and %s...\n", trainPath.string().c_str(), valPath.string().c_str());

        auto trainSet = Dataset(trainPath, cfg.maxSeqLen, device.get());
        auto valSet   = Dataset(valPath,   cfg.maxSeqLen, device.get());

        std::printf("Train samples: %zu\n", trainSet.size().value_or(0));
        std::printf("Val samples: %zu\n",   valSet.size().value_or(0));

        auto model = Gpt(cfg);

        const auto paramCount = [&]()
        {
            int64_t count = 0;
            for (const auto& param : model->parameters())
            {
                count += param.numel();
            }
            return count;
        }();

        std::printf("Model parameters: %lld\n", static_cast<long long int>(paramCount));

        ExperimentLog expLog{cfg.logFile};
        MetricsLog metricsLog{cfg.metricsFile};

        std::printf("Starting training for up to %zu seconds...\n", cfg.trainBudgetSecs);
        std::printf("Live metrics -> %s  (run TensorBoardWatch.py in another terminal)\n", cfg.metricsFile.c_str());

        size_t finalStep = 0;
        auto eval = std::make_unique<Evaluator>(cfg, valSet, device.get());

        Trainer trainer(cfg, std::move(model), trainSet, std::move(eval), metricsLog, device.get());

        const double finalBpb = trainer.run([&finalStep](const size_t step, double /*bpb*/)
        {
            finalStep = step;
        });

        ExperimentRecord record;
        record.timestamp  = currentTimestamp();
        record.deviceDesc = device.describe();
        record.valBpb     = finalBpb;
        record.steps      = finalStep;
        record.budgetSecs = cfg.trainBudgetSecs;

        expLog.append(record);  // 3. use expLog, not log

        std::printf("Training complete. Final val_bpb: %.6f\n", finalBpb);
        std::printf("Experiment record logged to %s\n", cfg.logFile.c_str());

        auto ckptPath = checkpointDir / ("model_step" + std::to_string(finalStep) + ".pt");
        torch::save(trainer.model(), ckptPath.string());
        std::printf("Model checkpoint saved to %s\n", ckptPath.string().c_str());

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Runtime error: " << e.what() << std::endl;
        std::cerr << "Hint: ensure --data-dir contains both train.bin and val.bin." << std::endl;
        return EXIT_FAILURE;
    }
}