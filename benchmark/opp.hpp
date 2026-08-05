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
   ::opp::encode(input, output);
}
//---------------------------------------------------------------------------
inline void decompress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   ::opp::decode(input, output);
}
//---------------------------------------------------------------------------
}
