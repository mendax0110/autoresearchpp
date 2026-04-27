#include "../include/Metricslog.h"
#include <iomanip>
#include <sstream>
#include <stdexcept>

using namespace autoresearch;

MetricsLog::MetricsLog(const std::filesystem::path& path)
    : m_stream(path, std::ios::app)
{
    if (!m_stream)
    {
        throw std::runtime_error("Failed to open metrics log file for appending: " + path.string());
    }
}

void MetricsLog::append(const MetricEntry& entry)
{
    m_stream << toJson(entry) << std::endl;
    m_stream.flush();
}

std::string MetricsLog::toJson(const MetricEntry& entry)
{
    std::ostringstream oss;

    oss << std::fixed << std::setprecision(6);
    oss << '{'
        << "\"step\":" << entry.step << ','
        << "\"elapsed_s\":" << entry.elapsedS << ','
        << "\"train_loss\":" << entry.trainLoss << ','
        << "\"val_bpb\":" << entry.valBpb
        << '}';
    return oss.str();
}
