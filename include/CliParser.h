#pragma once

#include "Config.h"
#include <span>
#include <string_view>

namespace autoresearch
{
    /// @brief Parses command-line args into a config. \class CliParser
    class CliParser
    {
    public:
        /**
         * @brief Parses argv into a config
         * @param args Span over the raw args strings
         * @return Populated Config.
         * @throws std::runtime_error if any args are malformed or invalid.
         */
        [[nodiscard]] static Config parse(std::span<const char* const> args);

        /**
         * @brief Prints usage to stdout.
         * @param programName The name to display in the usage line.
         */
        static void printUsage(std::string_view programName);
    };
}