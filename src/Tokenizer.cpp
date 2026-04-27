#include "../include/Tokenizer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace autoresearch;

Tokenizer::Tokenizer(const std::filesystem::path& vocabPath)
{
    loadVocab(vocabPath);
}

void Tokenizer::loadVocab(const std::filesystem::path& path)
{
    std::ifstream file(path);
    if (!file)
    {
        throw std::runtime_error("Failed to open vocab file: " + path.string());
    }

    std::string line;
    int rank = 0;

    while (std::getline(file, line))
    {
        if (line.empty()) { continue; }
        m_vocab.push_back(line);
        m_mergeRanks[line] = rank++;
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const
{
    std::vector<int32_t> ids;
    size_t pos = 0;

    while (pos < text.size())
    {
        size_t bestLen = 1;
        int bestId = -1;

        for (size_t len = text.size() - pos; len >= 1; --len)
        {
            std::string candidate = text.substr(pos, len);
            auto it = m_mergeRanks.find(candidate);
            if (it != m_mergeRanks.end())
            {
                bestLen = len;
                bestId = it->second;
                break;
            }
        }

        if (bestId == -1)
        {
            // Unknown byte: encode as raw byte token if available, esle we skip.
            pos++;
            continue;
        }

        ids.push_back(static_cast<int32_t>(bestId));
        pos += bestLen;
    }

    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const
{
    std::string result;
    result.reserve(ids.size() * 4);

    for (const auto id : ids)
    {
        if (id >= 0 && static_cast<size_t>(id) < m_vocab.size())
        {
            result += m_vocab[static_cast<size_t>(id)];
        }
    }

    return result;
}

size_t Tokenizer::vocabSize() const noexcept
{
    return m_vocab.size();
}
