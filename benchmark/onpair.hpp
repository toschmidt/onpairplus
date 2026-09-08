#pragma once
//---------------------------------------------------------------------------
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>
#include <onpair.h>
#include <onpair16.h>
//---------------------------------------------------------------------------
// The reference OnPair implementation (Gargiulo and Venturini, arXiv:2508.02280), the algorithm
// OnPair+ builds on. It compresses a *collection* of strings for random access; the whole input is
// passed as a single string here, matching how every other compressor is measured.
//
// The library keeps its result in three vectors, so they are serialized into one blob:
//   [u16 token stream][dictionary bytes][u32 token boundaries][Footer]
// This mirrors the framing token-vldb2026 uses for the same library and counts the dictionary
// against the compressed size, exactly as opp's own stream format does.
//
// onpair   - unbounded symbol length
// onpair16 - symbols capped at 16 bytes, the direct counterpart of opp's MAX_SYMBOL_LENGTH
//
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace bench::onpair {
//---------------------------------------------------------------------------
/// Both variants decompress by copying a fixed 16 bytes per token, so the dictionary and the
/// output buffer need that much slack past their last byte
inline constexpr size_t PADDING = 16;
//---------------------------------------------------------------------------
struct Footer {
   uint64_t uncompressedSize;
   uint64_t dictionarySize;
   uint64_t tokenBoundariesSize;
};
//---------------------------------------------------------------------------
template <class OnPairImpl>
void compressWith(std::span<const std::byte> input, std::vector<std::byte>& output) {
   OnPairImpl op(1, input.size());
   const std::vector<size_t> endPositions = {0, input.size()};
   op.compress_bytes(reinterpret_cast<const uint8_t*>(input.data()), endPositions);

   const Footer footer{input.size(), op.dictionary.size(), op.token_boundaries.size() * sizeof(uint32_t)};
   const size_t tokenBytes = op.compressed_data.size() * sizeof(uint16_t);
   output.resize(tokenBytes + footer.dictionarySize + footer.tokenBoundariesSize + sizeof(Footer));

   auto* writer = output.data();
   std::memcpy(writer, op.compressed_data.data(), tokenBytes);
   writer += tokenBytes;
   std::memcpy(writer, op.dictionary.data(), footer.dictionarySize);
   writer += footer.dictionarySize;
   std::memcpy(writer, op.token_boundaries.data(), footer.tokenBoundariesSize);
   writer += footer.tokenBoundariesSize;
   std::memcpy(writer, &footer, sizeof(Footer));
}
//---------------------------------------------------------------------------
template <class OnPairImpl>
void decompressWith(std::span<const std::byte> input, std::vector<std::byte>& output) {
   if (input.size() < sizeof(Footer)) throw std::runtime_error("onpair: input too small");
   Footer footer;
   std::memcpy(&footer, input.data() + input.size() - sizeof(Footer), sizeof(Footer));
   if (footer.dictionarySize + footer.tokenBoundariesSize + sizeof(Footer) > input.size())
      throw std::runtime_error("onpair: corrupt footer");
   const size_t tokenBytes = input.size() - sizeof(Footer) - footer.dictionarySize - footer.tokenBoundariesSize;

   OnPairImpl op;
   const auto* reader = input.data();
   op.compressed_data.resize(tokenBytes / sizeof(uint16_t));
   std::memcpy(op.compressed_data.data(), reader, tokenBytes);
   reader += tokenBytes;
   op.dictionary.assign(footer.dictionarySize + PADDING, 0); // slack for the fixed-width copy
   std::memcpy(op.dictionary.data(), reader, footer.dictionarySize);
   reader += footer.dictionarySize;
   op.token_boundaries.resize(footer.tokenBoundariesSize / sizeof(uint32_t));
   std::memcpy(op.token_boundaries.data(), reader, footer.tokenBoundariesSize);

   output.resize(footer.uncompressedSize + PADDING);
   op.decompress_all(reinterpret_cast<uint8_t*>(output.data()));
   output.resize(footer.uncompressedSize);
}
//---------------------------------------------------------------------------
inline constexpr const char* name = "onpair";
inline void compress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   compressWith<::OnPair>(input, output);
}
inline void decompress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   decompressWith<::OnPair>(input, output);
}
//---------------------------------------------------------------------------
}
//---------------------------------------------------------------------------
namespace bench::onpair16 {
//---------------------------------------------------------------------------
inline constexpr const char* name = "onpair16";
inline void compress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   ::bench::onpair::compressWith<::OnPair16>(input, output);
}
inline void decompress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   ::bench::onpair::decompressWith<::OnPair16>(input, output);
}
//---------------------------------------------------------------------------
}
