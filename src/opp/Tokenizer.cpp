#include "opp/Tokenizer.hpp"
#include <algorithm>
#include <bit>
#include <cassert>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
bool Tokenizer::isPrefix(const uint64_t text, const uint64_t prefix, const size_t textLen, const size_t prefixLen)
// Check whether the `prefixLen`-byte prefix is a byte-prefix of the `textLen`-byte text
{
   return prefixLen <= textLen && static_cast<size_t>(std::countr_zero(text ^ prefix) >> 3) >= prefixLen;
}
//---------------------------------------------------------------------------
bool Tokenizer::insert(const uint8_t* data, const size_t length, const uint16_t id) {
   assert(length >= 1 && length <= 16);
   if (length == 1) return true; // single-byte symbols are implicit (identity ids 0..255)

   const uint64_t value = loadPartial(data, std::min<size_t>(length, 8));
   if (length == 2) {
      // Direct-lookup tail array; a slot still holding the single-byte default is free
      auto& slot = shortTokenArray[static_cast<uint16_t>(value)];
      if (slot < 256) slot = id;
      return true;
   }
   if (length <= 4) {
      tokenMaps4[length - 3].try_emplace(static_cast<uint32_t>(value) << ((4 - length) * 8), id);
      return true;
   }
   if (length <= 8) {
      tokenMaps8[length - 5].try_emplace(value << ((8 - length) * 8), id);
      return true;
   }
   // Long symbol

   // Compute prefix
   auto& bucket = longLookupTable[value];
   if (bucket.size() >= MAX_BUCKET_SIZE) return false;

   // Compute suffix
   const size_t suffixLen = length - 8;
   const uint64_t suffix = loadPartial(data + 8, suffixLen);
   bucket.emplace_back(suffix, static_cast<uint8_t>(suffixLen), id);

   // Sort by suffix length in descending order
   std::sort(bucket.begin(), bucket.end(), [](const auto& a, const auto& b) {
      return std::get<1>(b) < std::get<1>(a);
   });

   return true;
}
//---------------------------------------------------------------------------
std::optional<std::pair<uint16_t, size_t>> Tokenizer::findLongestMatch(const uint8_t* data, const size_t length) const {
   // Long match handling (8 < x <= 16 bytes), using the dynamic table
   if (length > 8) {
      const size_t suffixLen = std::min(length, size_t{16}) - 8;
      const uint64_t prefix = loadPartial(data, 8);
      const uint64_t suffix = loadPartial(data + 8, suffixLen);

      if (const auto bucketIt = longLookupTable.find(prefix); bucketIt != longLookupTable.end()) {
         for (const auto& entry : bucketIt->second) {
            if (const auto& [entrySuffix, entrySuffixLen, entryId] = entry; isPrefix(suffix, entrySuffix, suffixLen, entrySuffixLen)) {
               return std::make_pair(entryId, 8 + entrySuffixLen);
            }
         }
      }
   }
   // Probe the per-length tables from the longest possible length down
   const uint64_t value = loadPartial(data, std::min(length, size_t{8}));
   for (size_t l = std::min(length, size_t{8}); l >= 5; --l) {
      const uint64_t key = (value & MASKS[l]) << ((8 - l) * 8);
      if (const auto it = tokenMaps8[l - 5].find(key); it != tokenMaps8[l - 5].end())
         return std::make_pair(it->second, l);
   }
   for (size_t l = std::min(length, size_t{4}); l >= 3; --l) {
      const auto key = static_cast<uint32_t>(value & MASKS[l]) << ((4 - l) * 8);
      if (const auto it = tokenMaps4[l - 3].find(key); it != tokenMaps4[l - 3].end())
         return std::make_pair(it->second, l);
   }
   // The 2-byte/1-byte tail: one array load resolves either a 2-byte symbol or the single byte
   if (length >= 2) {
      const uint16_t id = shortTokenArray[static_cast<uint16_t>(value)];
      return std::make_pair(id, static_cast<size_t>(1) + (id > 255));
   }
   return std::make_pair(static_cast<uint16_t>(value), size_t{1});
}
//---------------------------------------------------------------------------
void Tokenizer::remapIds(const std::span<const uint16_t> remap)
// Rewrite every stored token id through the remap table (old id -> new id)
{
   for (auto& slot : shortTokenArray) slot = remap[slot];
   for (auto& map : tokenMaps4)
      for (auto& [key, id] : map) id = remap[id];
   for (auto& map : tokenMaps8)
      for (auto& [key, id] : map) id = remap[id];
   for (auto& [prefix, bucket] : longLookupTable)
      for (auto& entry : bucket) std::get<2>(entry) = remap[std::get<2>(entry)];
}
//---------------------------------------------------------------------------
void Tokenizer::finalize(const uint16_t firstTrainedId)
// Build the static compression tables from the dynamic ones; call after remapIds()
{
   // The 3-byte map: the dynamic keys (symbol << 8) already match the lookup format. The map is
   // lossy and keeps the first inserted key of a full slot, so insert by descending importance:
   // trained symbols (frequent by construction) first, then the n-gram trigrams in level order
   // (most frequent characters first), so a collision drops the least valuable symbol.
   {
      Vector<std::pair<uint32_t, uint16_t>> symbols3(memory);
      symbols3.reserve(tokenMaps4[0].size());
      for (const auto& [key, id] : tokenMaps4[0])
         if (key != 0) symbols3.emplace_back(key, id); // key 0 is the map's empty sentinel
      std::sort(symbols3.begin(), symbols3.end(), [&](const auto& a, const auto& b) {
         const bool aTrained = a.second >= firstTrainedId, bTrained = b.second >= firstTrainedId;
         return aTrained != bTrained ? aTrained : a.second < b.second;
      });

      Vector<uint32_t> keys(memory);
      keys.reserve(symbols3.size());
      for (const auto& [key, id] : symbols3) keys.push_back(key);
      tokenMap3.initialize(keys);
      for (const auto& [key, id] : symbols3) tokenMap3[key] = id;
   }

   // The 4..8 byte map, keyed by the first 4 bytes, bucket entries sorted longest-first (ties
   // broken by id for deterministic output)
   {
      Vector<ShortTokenEntry> entries(memory);
      entries.reserve(tokenMaps4[1].size() + tokenMaps8[0].size() + tokenMaps8[1].size() + tokenMaps8[2].size() + tokenMaps8[3].size());
      for (const auto& [key, id] : tokenMaps4[1])
         entries.push_back({key, id, 4});
      for (unsigned l = 5; l <= 8; ++l)
         for (const auto& [key, id] : tokenMaps8[l - 5])
            entries.push_back({key >> ((8 - l) * 8), id, l});
      std::sort(entries.begin(), entries.end(), [](const ShortTokenEntry& a, const ShortTokenEntry& b) {
         return a.length != b.length ? a.length > b.length : a.id < b.id;
      });

      Vector<uint32_t> keys(memory);
      keys.reserve(entries.size());
      for (const auto& entry : entries) keys.push_back(static_cast<uint32_t>(entry.text));
      shortTokenMap.initialize(keys);
      for (const auto& entry : entries) shortTokenMap[static_cast<uint32_t>(entry.text)] = entry;
      if (!shortTokenMap.empty())
         shortTokenMap.setDefault(ShortTokenEntry{.text = 1, .id = 0, .length = 0}); // never matches
   }

   // The long symbol map (9..16 bytes), keyed by the first 8 bytes, bucket entries sorted
   // longest-first (ties broken by id for deterministic output)
   {
      Vector<LongTokenEntry> entries(memory);
      for (const auto& [prefix, bucket] : longLookupTable)
         for (const auto& [suffix, suffixLen, id] : bucket)
            entries.push_back({prefix, suffix, id, 8u + suffixLen});
      std::sort(entries.begin(), entries.end(), [](const LongTokenEntry& a, const LongTokenEntry& b) {
         return a.length != b.length ? a.length > b.length : a.id < b.id;
      });

      Vector<uint64_t> keys(memory);
      keys.reserve(entries.size());
      for (const auto& entry : entries) keys.push_back(entry.prefix);
      longTokenMap.initialize(keys);
      for (const auto& entry : entries) longTokenMap[entry.prefix] = entry;
      if (!longTokenMap.empty())
         longTokenMap.setDefault(LongTokenEntry{.prefix = 0, .postfix = 1, .id = 0, .length = 8}); // never matches
   }
}
//---------------------------------------------------------------------------
}
