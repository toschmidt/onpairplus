#include "OnPairPlus.hpp"
#include "opp/Config.hpp"
#include "opp/Tokenizer.hpp"
#include "util/Infra.hpp"
#include "catch2/catch_test_macros.hpp"
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
using namespace opp;
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
std::vector<std::byte> toBytes(const std::string& s) {
   std::vector<std::byte> data(s.size());
   std::memcpy(data.data(), s.data(), s.size());
   return data;
}
//---------------------------------------------------------------------------
/// Roundtrip an input through encode/decode and check the output matches; returns the compressed size
size_t roundtrip(const std::vector<std::byte>& input) {
   std::vector<std::byte> compressed, restored;
   opp::encode(input, compressed);
   opp::decode(compressed, restored);
   REQUIRE(restored == input);
   return compressed.size();
}
//---------------------------------------------------------------------------
/// Build a matcher holding the given symbols (with ids 256, 257, ...) and its compression tables
Tokenizer makeMatcher(const std::vector<std::string>& symbols) {
   Tokenizer matcher;
   uint16_t nextId = 256;
   for (const auto& symbol : symbols)
      REQUIRE(matcher.insert(reinterpret_cast<const uint8_t*>(symbol.data()), symbol.size(), nextId++));

   std::vector<uint16_t> identity(65536);
   for (size_t i = 0; i < identity.size(); ++i) identity[i] = static_cast<uint16_t>(i);
   matcher.remapIds(identity);
   matcher.finalize();
   return matcher;
}
//---------------------------------------------------------------------------
/// Match a string through the dense path (guaranteeing 16 readable bytes)
std::pair<uint16_t, size_t> matchDense(const Tokenizer& matcher, const std::string& input) {
   const std::string padded = input + std::string(16, '\0');
   return matcher.match(reinterpret_cast<const uint8_t*>(padded.data()));
}
//---------------------------------------------------------------------------
}
//---------------------------------------------------------------------------
TEST_CASE("RoundtripEdgeSizes") {
   // Empty, single bytes, and every size around the matcher's dense/tail boundary
   roundtrip({});
   for (size_t size : {1ul, 2ul, 3ul, 7ul, 8ul, 9ul, 15ul, 16ul, 17ul, 31ul, 255ul, 256ul, 257ul}) {
      std::vector<std::byte> input(size);
      for (size_t i = 0; i < size; ++i) input[i] = static_cast<std::byte>('a' + i % 26);
      roundtrip(input);
   }
   // Sizes around the training chunk boundary
   for (size_t size : {TRAIN_CHUNK_SIZE - 1, TRAIN_CHUNK_SIZE, TRAIN_CHUNK_SIZE + 1, 3 * TRAIN_CHUNK_SIZE + 17}) {
      std::vector<std::byte> input(size);
      for (size_t i = 0; i < size; ++i) input[i] = static_cast<std::byte>("the quick brown fox "[i % 20]);
      roundtrip(input);
   }
}
//---------------------------------------------------------------------------
TEST_CASE("RoundtripAllByteValues") {
   // Every byte value, including zeros, in ascending order and repeated
   std::vector<std::byte> input(256 * 16);
   for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<std::byte>(i / 16);
   roundtrip(input);
}
//---------------------------------------------------------------------------
TEST_CASE("RoundtripRandom") {
   // Incompressible data must roundtrip and stay within the worst-case bound (2 bytes per input
   // byte for the token stream, plus the size prefix and dictionary)
   std::mt19937 g(1234);
   std::vector<std::byte> input(1 << 20);
   for (auto& b : input) b = static_cast<std::byte>(g());
   const size_t compressed = roundtrip(input);
   REQUIRE(compressed <= sizeof(uint64_t) + MAX_DICTIONARY_SIZE + 2 * input.size());
}
//---------------------------------------------------------------------------
TEST_CASE("RoundtripText") {
   // Repetitive text must compress well
   std::string text;
   const std::string words[] = {"http://example.com/", "banana", "compression", "42", " ", "\n"};
   std::mt19937 g(42);
   while (text.size() < (1u << 20)) text += words[g() % 6];
   const size_t compressed = roundtrip(toBytes(text));
   REQUIRE(compressed < text.size() / 3);
}
//---------------------------------------------------------------------------
TEST_CASE("Determinism") {
   // Encoding the same input twice must produce identical bytes
   std::mt19937 g(7);
   std::vector<std::byte> input(200000);
   for (auto& b : input) b = static_cast<std::byte>('a' + g() % 16);
   std::vector<std::byte> first, second;
   opp::encode(input, first);
   opp::encode(input, second);
   REQUIRE(first == second);
}
//---------------------------------------------------------------------------
TEST_CASE("StreamFormat") {
   // The stream starts with the uncompressed size
   const auto input = toBytes("hello world, hello world, hello world");
   std::vector<std::byte> compressed;
   opp::encode(input, compressed);
   uint64_t storedSize;
   std::memcpy(&storedSize, compressed.data(), sizeof(storedSize));
   REQUIRE(storedSize == input.size());
}
//---------------------------------------------------------------------------
TEST_CASE("DecodeErrors") {
   // Too-small inputs must throw instead of crashing
   for (size_t size : {0ul, 1ul, 7ul, 8ul, 20ul}) {
      std::vector<std::byte> bogus(size), output;
      REQUIRE_THROWS(opp::decode(bogus, output));
   }
}
//---------------------------------------------------------------------------
TEST_CASE("Matcher") {
   // The dense and the length-limited match agree with the dynamic training-time matcher
   const std::vector<std::string> symbols = {"ab", "abc", "abcd", "abcde", "abcdefgh", "abcdefghi", "abcdefghijklmnop", "xy", "xyz"};
   const auto matcher = makeMatcher(symbols);

   const std::string inputs[] = {"abcdefghijklmnopqrst", "abcdefghij", "abcdefgh!", "abcdez", "abcz", "abz", "az", "xyz!", "z...", "a"};
   for (const auto& input : inputs) {
      const std::string padded = input + std::string(16, '\0');
      const auto* data = reinterpret_cast<const uint8_t*>(padded.data());

      const auto expected = *matcher.findLongestMatch(data, input.size());
      REQUIRE(matcher.match(data) == expected);
      if (input.size() < 16)
         REQUIRE(matcher.match(reinterpret_cast<const uint8_t*>(input.data()), input.size()) == expected);
   }

   // The longest match wins: "abcdefghijklmnop" (16 bytes) over all its prefixes
   REQUIRE(matchDense(matcher, "abcdefghijklmnop").second == 16);
   // Unknown bytes fall back to the single-byte symbol with the identity id
   REQUIRE(matchDense(matcher, "Z") == std::pair<uint16_t, size_t>{'Z', 1});
}
//---------------------------------------------------------------------------
// Zero bytes are special in the compression tables: key 0 is the lossy 3-byte map's empty
// sentinel, entries < 256 mark free slots in the 2-byte tail array, and past the input's end
// the loads are zero-extended. These tests pin the intended behavior down.
//---------------------------------------------------------------------------
TEST_CASE("MatcherZeroBytes") {
   using Match = std::pair<uint16_t, size_t>;
   const std::string z2("\0\0", 2);
   const std::string za("\0a", 2);
   const std::string z3("\0\0\0", 3);
   const std::string a0b("a\0\0b", 4);
   const std::string long0("abcdefgh\0\0wxyz", 14);

   SECTION("two-byte symbols with zero bytes live in the tail array") {
      const auto matcher = makeMatcher({z2, za});
      REQUIRE(matchDense(matcher, std::string("\0\0\0\0", 4)) == Match{256, 2});
      REQUIRE(matchDense(matcher, std::string("\0ab", 3)) == Match{257, 2});
      // A zero followed by an unknown byte matches the single-byte symbol with id 0
      REQUIRE(matchDense(matcher, std::string("\0q", 2)) == Match{0, 1});
      // A lone zero at the input's end zero-extends and over-matches the 2-byte "\0\0" symbol
      // (the decoder's truncation removes the surplus zero byte again)
      const std::string lonely("\0q", 2);
      REQUIRE(matcher.match(reinterpret_cast<const uint8_t*>(lonely.data()), 1) == Match{256, 2});
   }

   SECTION("the all-zero 3-byte symbol collides with the lossy map's sentinel and is skipped") {
      const auto matcher = makeMatcher({z3});
      // Never a bogus 3-byte match: zeros fall back to the single-byte symbol (id 0)
      REQUIRE(matchDense(matcher, std::string("\0\0\0\0\0", 5)) == Match{0, 1});
   }

   SECTION("non-degenerate 3-byte symbols with embedded zeros work") {
      const std::string x0y("x\0y", 3);
      const auto matcher = makeMatcher({x0y});
      REQUIRE(matchDense(matcher, x0y + "...") == Match{256, 3});
   }

   SECTION("4..8 and long symbols with embedded zeros work") {
      const auto matcher = makeMatcher({a0b, long0});
      REQUIRE(matchDense(matcher, a0b + "....") == Match{256, 4});
      REQUIRE(matchDense(matcher, long0 + "..") == Match{257, 14});
   }

   SECTION("a symbol with trailing zeros may over-match past the input's end") {
      const std::string ab00("ab\0\0", 4);
      const auto matcher = makeMatcher({ab00});
      // Only "ab" is readable; the zero-extended load matches the 4-byte symbol. This is the
      // intended over-match: the decoder's truncation removes the surplus zero bytes again.
      REQUIRE(matcher.match(reinterpret_cast<const uint8_t*>(ab00.data()), 2) == Match{256, 4});
   }
}
//---------------------------------------------------------------------------
TEST_CASE("RoundtripZeroBytes") {
   SECTION("all-zero input") {
      roundtrip(std::vector<std::byte>(100000, std::byte{0}));
   }
   SECTION("zero-separated records, input ends mid-pattern") {
      // Zero-heavy repetitive data trains symbols with embedded/trailing zeros; ending the input
      // inside a repeated pattern exercises the zero-extension over-match at the tail
      std::string text;
      while (text.size() < 300000) text += std::string("record\0\0\0\0", 10);
      text += "record"; // truncated final record
      roundtrip(toBytes(text));
   }
   SECTION("random data with frequent zeros") {
      std::mt19937 g(99);
      std::vector<std::byte> input(1 << 18);
      for (auto& b : input) b = static_cast<std::byte>(g() % 4 == 0 ? 0 : g() % 256);
      roundtrip(input);
   }
   SECTION("all 65536 two-byte combinations") {
      // Contains every possible 2-byte value, including all zero-containing ones
      std::vector<std::byte> input(2 * 65536);
      for (size_t i = 0; i < 65536; ++i) {
         input[2 * i] = static_cast<std::byte>(i & 0xFF);
         input[2 * i + 1] = static_cast<std::byte>(i >> 8);
      }
      roundtrip(input);
   }
}
