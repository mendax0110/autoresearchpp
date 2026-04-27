#include "../include/Tokenizer.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

using namespace autoresearch;

namespace
{
    std::filesystem::path makeTempVocabPath()
    {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return std::filesystem::temp_directory_path() / ("tokenizer-test-" + std::to_string(stamp) + ".vocab");
    }
}

TEST(TokenizerTest, RoundTripUsesLongestMatch)
{
    const auto vocabPath = makeTempVocabPath();
    std::ofstream out(vocabPath);
    ASSERT_TRUE(out.is_open());
    out << "a\n";
    out << "b\n";
    out << "ab\n";
    out.close();

    const Tokenizer tokenizer(vocabPath);
    const auto ids = tokenizer.encode("ab");

    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 2);
    EXPECT_EQ(tokenizer.decode(ids), "ab");

    std::filesystem::remove(vocabPath);
}

TEST(TokenizerTest, UnknownCharactersAreSkipped)
{
    const auto vocabPath = makeTempVocabPath();
    std::ofstream out(vocabPath);
    ASSERT_TRUE(out.is_open());
    out << "a\n";
    out.close();

    const Tokenizer tokenizer(vocabPath);
    const auto ids = tokenizer.encode("az");

    ASSERT_EQ(ids.size(), 1u);
    EXPECT_EQ(ids[0], 0);
    EXPECT_EQ(tokenizer.decode({0, 99}), "a");

    std::filesystem::remove(vocabPath);
}
