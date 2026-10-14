#pragma once
//---------------------------------------------------------------------------
#include <cstddef>
#include <cstdint>
#include <utility>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
/// wyhash-style hasher for integers and integer pairs
struct Hasher {
   /// Pseudorandom "secret" constants from wyhash
   static constexpr uint64_t k0 = 0xa0761d6478bd642full, k1 = 0xe7037ed1a0b428dbull, k4 = 0x589965cc75374cc3ull;
   /// Full 128 bit multiplication folded into 64bit
   static constexpr uint64_t mulFold(uint64_t a, uint64_t b) noexcept {
      auto m = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
      return static_cast<uint64_t>(m >> 64) ^ static_cast<uint64_t>(m);
   }
   /// Hash two values
   static constexpr uint64_t hashTwo(uint64_t v1, uint64_t v2) noexcept {
      return mulFold(k4 ^ 16, mulFold(v1 ^ k1, v2 ^ k0));
   }
   /// Function-call operators
   constexpr std::size_t operator()(uint64_t v) const noexcept { return hashTwo(v, v); }
   template <typename A, typename B>
   constexpr std::size_t operator()(const std::pair<A, B>& p) const noexcept {
      return hashTwo(static_cast<uint64_t>(p.first), static_cast<uint64_t>(p.second));
   }
};
//---------------------------------------------------------------------------
/// A fast multiplicative hash for integer keys with well-mixed high bits
inline uint64_t fasthash(const uint64_t key) noexcept {
   const auto m = static_cast<unsigned __int128>(key) * 0x2127'599b'f432'5c37ull;
   return static_cast<uint64_t>(m >> 64) ^ static_cast<uint64_t>(m);
}
//---------------------------------------------------------------------------
/// The number of bits needed to address at least `a` slots
constexpr uint32_t log2ceil(const uint64_t a) noexcept {
   return a <= 1 ? 1 : 64 - static_cast<uint32_t>(__builtin_clzll(a - 1));
}
//---------------------------------------------------------------------------
}
