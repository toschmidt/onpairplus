#pragma once
//---------------------------------------------------------------------------
#include "util/Hash.hpp"
#include <cassert>
#include <cstdint>
#include <span>
#include <vector>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
/// A static multi-value hash map: initialize() counts the keys and builds a directory (counting
/// sort), access() then appends each key's value into its bucket in call order. find() returns
/// the bucket range; the four padding entries past the last bucket allow unrolled probing.
/// Ported from the SGTT tokenizer.
template <class Key, class Value>
class MultiHashMap {
   /// The right-shift mapping a hash to a slot
   uint32_t shift = 63;
   /// The directory (bucket boundaries into values)
   std::vector<uint32_t> directory;
   /// The values
   std::vector<Value> values;
   /// The per-bucket insert counters
   std::vector<uint32_t> insertCounter;

   /// Map a key to its slot
   uint64_t slotOf(const Key key) const noexcept { return fasthash(static_cast<uint64_t>(key)) >> shift; }

   public:
   /// Constructor
   MultiHashMap() { directory.resize(2, 0); }

   /// Whether the map holds no entries
   bool empty() const noexcept { return values.empty(); }

   /// Size the table for the given keys (call once, before any access)
   void initialize(const std::span<const Key> keys) {
      assert(shift == 63);
      if (keys.empty()) return;

      const unsigned sizeLog2 = log2ceil(keys.size()) + 2;
      const size_t tableSize = (1ull << sizeLog2) + 3;
      shift = 64 - sizeLog2;

      directory.assign(tableSize + 1, 0);
      values.resize(keys.size() + 4);
      insertCounter.assign(tableSize, 0);

      for (const auto key : keys) ++insertCounter[slotOf(key)];
      for (size_t i = 1; i <= tableSize; ++i) {
         directory[i] = directory[i - 1] + insertCounter[i - 1];
         insertCounter[i - 1] = 0;
      }
   }
   /// Append the value of a key to its bucket (in call order)
   Value& access(const Key key) {
      const auto slot = slotOf(key);
      const auto offset = insertCounter[slot]++;
      assert(directory[slot] + offset < directory[slot + 1]);
      return values[directory[slot] + offset];
   }
   /// Access operator
   Value& operator[](const Key key) { return access(key); }
   /// Fill the padding entries past the last bucket (probed by unrolled scans)
   void setDefault(const Value& value) {
      for (size_t i = values.size() - 4; i < values.size(); ++i) values[i] = value;
   }

   /// Find the bucket range of a key
   std::pair<const Value*, const Value*> find(const Key key) const noexcept {
      const auto slot = slotOf(key);
      return {values.data() + directory[slot], values.data() + directory[slot + 1]};
   }
};
//---------------------------------------------------------------------------
}
