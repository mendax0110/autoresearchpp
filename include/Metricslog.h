#pragma once

#include <filesystem>
#include <fstream>
#include <string>

namespace autoresearch
{
    /// @brief Single record of training metrics to be logged. \struct MetricEntry
    struct MetricEntry
    {
        size_t step = 0;
        double elapsedS = 0.0;
        float trainLoss = 0.0f;
        double valBpb = 0.0;
    };

    /// @brief Appends metric entries as JSONL for consumption by external tools. \class MetricsLog
    class MetricsLog
    {
    public:
        /**
         * @brief Opens or creates the metrics file.
         * @param path Path to the .jsonl metrics file
         * @throws std::runtime_error if the file cannot be opened for appending.
         */
        explicit MetricsLog(const std::filesystem::path& path);

        ~MetricsLog() = default;

        MetricsLog(const MetricsLog&) = delete;
        MetricsLog& operator=(const MetricsLog&) = delete;
        MetricsLog(MetricsLog&&) = default;
        MetricsLog& operator=(MetricsLog&&) = default;

        /**
         * @brief Appends a metric entry as a JSON object on a new line.
         * @param entry The mertic values for this step.
         */
        void append(const MetricEntry& entry);

    private:
        std::ofstream m_stream;

        /**
         * @brief Converts a MetricEntry to a JSON string.
         * @param entry The metric entry to convert.
         * @return A JSON-formatted string representing the metric entry.
         */
        static std::string toJson(const MetricEntry& entry);
    };
}