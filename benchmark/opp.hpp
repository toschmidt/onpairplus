#pragma once
//---------------------------------------------------------------------------
#include <OnPairPlus.hpp>
#include <span>
#include <vector>
//---------------------------------------------------------------------------
// OnPair+ adapter (the compressor under test, from ../src)
//
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace bench::opp {
//---------------------------------------------------------------------------
inline constexpr const char* name = "opp";
//---------------------------------------------------------------------------
inline void compress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   // OnPair+ encodes into an opp::Vector (a std::pmr::vector), the benchmark keeps std::vector
   ::opp::Vector<std::byte> encoded;
   ::opp::encode(input, encoded);
   output.assign(encoded.begin(), encoded.end());
}
//---------------------------------------------------------------------------
inline void decompress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   // Decode straight into the benchmark's buffer (16 bytes of headroom for the decoder)
   const size_t size = ::opp::decodedSize(input);
   output.resize(size + 16);
   ::opp::decodeInto(input, output);
   output.resize(size);
}
//---------------------------------------------------------------------------
}
