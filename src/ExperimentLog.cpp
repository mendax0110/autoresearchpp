#include "../include/ExperimentLog.h"
#include <iomanip>
#include <sstream>
#include <stdexcept>

using namespace autoresearch;

ExperimentLog::ExperimentLog(const std::filesystem::path& path)
    : m_stream(path, std::ios::app)
{
    if (!m_stream)
    {
        throw std::runtime_error("Failed to open log file for appending: " + path.string());
    }
}

void ExperimentLog::append(const ExperimentRecord& record)
{
    m_stream << toJson(record) << std::endl;
    m_stream.flush();
}

std::string ExperimentLog::toJson(const ExperimentRecord& record)
{
    auto escape = [](const std::string& s) -> std::string
    {
        std::string out;
        out.reserve(s.size());
        for (const auto character : s)
        {
            if (character == '"') { out += "\\\""; }
            else if (character == '\\') { out += "\\\\"; }
            else { out += character; }
        }
        return out;
    };

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << '{'
        << R"("timestamp":")" << escape(record.timestamp) << "\","
        << R"("device":")" << escape(record.deviceDesc) << "\","
        << "\"val_bpb\":" << record.valBpb << ','
        << "\"steps\":" << record.steps << ','
        << "\"budget_secs\":" << record.budgetSecs << ','
        << R"("note":")" << escape(record.note) << '"'
        << '}';

    return oss.str();
}
