#pragma once

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace autoresearch
{
    /// @brief Experiment result written to JSONL log. \struct ExperimentRecord
    struct ExperimentRecord
    {
        std::string timestamp;
        std::string deviceDesc;
        double valBpb = 0.0;
        size_t steps = 0;
        size_t budgetSecs = 0;
        std::string note;
    };

    /// @brief Appends experiment results to a newline-delimited JSON log file. \class ExperimentLog
    class ExperimentLog
    {
    public:
        /**
         * @brief Opens or creates the log file.
         * @param path Path to the .json1 log file.
         * @throws std::runtime_error if the file cannot be opened for appending.
         */
        explicit ExperimentLog(const std::filesystem::path& path);

        ~ExperimentLog() = default;

        ExperimentLog(const ExperimentLog&) = delete;
        ExperimentLog& operator=(const ExperimentLog&) = delete;
        ExperimentLog(ExperimentLog&&) = default;
        ExperimentLog& operator=(ExperimentLog&&) = default;

        /**
         * @brief Appends a single record to the log.
         * @param record Experiment result to serialize.
         */
        void append(const ExperimentRecord& record);

    private:
        std::ofstream m_stream;

        /**
         * @brief Serializes an ExperimentRecord to a JSON string.
         * @param record The record to serialize.
         * @return A JSON string representation of the record.
         */
        static std::string toJson(const ExperimentRecord& record);
    };
}