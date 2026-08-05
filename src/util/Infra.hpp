#pragma once
//---------------------------------------------------------------------------
#include <cstring>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
using uint128_t = unsigned __int128;
//---------------------------------------------------------------------------
/// Load a value from a potentially unaligned address
template <typename T>
T unalignedLoad(const void* ptr) noexcept {
   T value;
   std::memcpy(&value, ptr, sizeof(T));
   return value;
}
//---------------------------------------------------------------------------
/// Store a value to a potentially unaligned address
template <typename T>
void unalignedStore(void* ptr, T value) noexcept { std::memcpy(ptr, &value, sizeof(T)); }
//---------------------------------------------------------------------------
}
