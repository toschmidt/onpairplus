#pragma once
//---------------------------------------------------------------------------
#include <span>
#include <vector>
#include <compressor/TokenizerCompressor.hpp>
//---------------------------------------------------------------------------
// SGTT tokenizer baseline from token-vldb2026: a fixed OpenAI-100k token table (patched to cover
// all byte values), no per-input training. The token table is a compile-time constant, so the
// compressor is stateless; its static lookup tables are built once on first use.
//
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace bench::tokenizer {
//---------------------------------------------------------------------------
inline constexpr const char* name = "tokenizer";
//---------------------------------------------------------------------------
/// The reference interface takes mutable spans but never writes through them
inline std::span<std::byte> mutableView(std::span<const std::byte> input) {
   return {const_cast<std::byte*>(input.data()), input.size()};
}
//---------------------------------------------------------------------------
inline void compress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   sgtt::compressor::TokenizerCompressor compressor;
   compressor.compress(mutableView(input), output);
}
//---------------------------------------------------------------------------
inline void decompress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   sgtt::compressor::TokenizerCompressor compressor;
   std::vector<std::vector<std::byte>> temporaryBuffers;
   compressor.decompress(mutableView(input), output, temporaryBuffers);
}
//---------------------------------------------------------------------------
}
