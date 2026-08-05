#pragma once
//---------------------------------------------------------------------------
#include "util/Hash.hpp"
#include <cassert>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
/// A static, lossy open-addressing hash map (two probe slots per key, no resizing): keys that do
/// not fit are silently dropped. Key 0 is the empty sentinel and must not be inserted or looked
/// up. Ported from the SGTT tokenizer.
template <class Key, class Value>
class LossyHashMap {
   /// The right-shift mapping a hash to a slot
   uint32_t shift = 63;
   /// The slots
   std::vector<std::pair<Key, Value>> values;
   /// Written by access() for dropped keys
   Value overflow{};

   /// Map a key to its slot
   uint64_t slotOf(const Key key) const noexcept { return fasthash(static_cast<uint64_t>(key)) >> shift; }

   public:
   /// Constructor
   LossyHashMap() { values.resize(3); }

   /// Size the table for the given keys (call once, before any access)
   void initialize(const std::span<const Key> keys) {
      assert(shift == 63);
      if (keys.empty()) return;

      const unsigned sizeLog2 = log2ceil(keys.size()) + 3;
      shift = 64 - sizeLog2;
      values.clear();
      values.resize((1ull << sizeLog2) + 3);
   }
   /// Access the entry of a key, claiming one of its two slots; dropped keys get a dummy slot
   Value& access(const Key key) {
      assert(key != Key{});
      const auto slot = slotOf(key);
      for (auto probe : {slot, slot + 1}) {
         if (values[probe].first == key || values[probe].first == Key{}) {
            values[probe].first = key;
            return values[probe].second;
         }
      }
      return overflow;
   }
   /// Access operator
   Value& operator[](const Key key) { return access(key); }

   /// Find the value of a key; returns nullptr if not present (or dropped)
   const Value* find(const Key key) const noexcept {
      const auto slot = slotOf(key);
      if (values[slot].first == key) [[likely]]
         return &values[slot].second;
      if (values[slot + 1].first == key) return &values[slot + 1].second;
      return nullptr;
   }
};
//---------------------------------------------------------------------------
}
