#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoresearch
{
    /// @brief Byte-pair encoding tokenizer loaded from a pre-built vocab file.
    class Tokenizer
    {
    public:
        /**
         * @brief Loads a BPE tokenizer from the given vocab file.
         * @param vocabPath Path to the .vocab file produced by prepare
         * @throws std::runtime_error if the vocab file cannot be read or is malformed.
         */
        explicit Tokenizer(const std::filesystem::path& vocabPath);

        /**
         * @brief Encodes a UTF-8 string into token IDs.
         * @param text The input text.
         * @return Sequence of token IDs
         */
        [[nodiscard]] std::vector<int32_t> encode(const std::string& text) const;

        /**
         * @brief Decodes a sequence of token IDs back into a UTF-8 string.
         * @param ids The input token IDs.
         * @return The decoded text.
         */
        [[nodiscard]] std::string decode(const std::vector<int32_t>& ids) const;

        /// @brief Returns the size of the vocabulary.
        [[nodiscard]] size_t vocabSize() const noexcept;

    private:
        std::vector<std::string> m_vocab;
        std::unordered_map<std::string, int> m_mergeRanks;

        /**
         * @brief Loads the vocabulary and merge ranks from the specified .vocab file.
         * @param path Path to the .vocab file.
         * @throws std::runtime_error if the file cannot be read or is malformed.
         */
        void loadVocab(const std::filesystem::path& path);
    };
}