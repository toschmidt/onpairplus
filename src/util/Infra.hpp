#pragma once
//---------------------------------------------------------------------------
#include <cstring>
#include <memory_resource>
#include <vector>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
using uint128_t = unsigned __int128;
//---------------------------------------------------------------------------
/// The vector of OnPair+: it allocates from the memory resource of its allocator, so every allocation
/// of an encode or decode goes through the resource the caller passes (the global heap by default)
template <class T>
using Vector = std::pmr::vector<T>;
/// The allocator OnPair+ hands to its containers
using Allocator = std::pmr::polymorphic_allocator<std::byte>;
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
