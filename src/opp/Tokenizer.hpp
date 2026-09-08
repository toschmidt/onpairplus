#pragma once
//---------------------------------------------------------------------------
#include "util/Hash.hpp"
#include "util/LossyHashMap.hpp"
#include "util/MultiHashMap.hpp"
#include "util/TrainMap.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
/// A compression-table entry for symbols of 4..8 bytes
struct ShortTokenEntry {
   /// The symbol bytes (little-endian, masked to its length)
   uint64_t text;
   /// The token id
   uint32_t id;
   /// The symbol length in bytes
   uint32_t length;
};
static_assert(sizeof(ShortTokenEntry) == 16);
//---------------------------------------------------------------------------
/// A compression-table entry for long symbols (9..16 bytes). Unlike the reference's exact-prefix
/// buckets, a MultiHashMap bucket can hold entries of several prefixes, so the prefix is stored
/// and compared too.
struct LongTokenEntry {
   /// The first 8 symbol bytes
   uint64_t prefix;
   /// The remaining symbol bytes (little-endian, masked to their length)
   uint64_t postfix;
   /// The token id
   uint32_t id;
   /// The symbol length in bytes
   uint32_t length;
};
static_assert(sizeof(LongTokenEntry) == 24);
//---------------------------------------------------------------------------
/// The Longest Prefix Matcher used to build the dictionary and tokenize the input.
///
/// Training uses dynamic per-length hash tables (findLongestMatch/insert). finalize() then builds
/// the static compression tables of the SGTT TokenizerCompressor, used by match()/matchLong():
/// a 65536-entry direct-lookup array resolving the 2-byte/1-byte tail without hashing, a lossy
/// two-probe map for 3-byte symbols, one multi hash map keyed by the first 4 bytes whose buckets
/// hold the 4..8 byte symbols longest-first, and a prefix-bucket table for long symbols (9..16).
class Tokenizer {
   /// Masks for byte sequences
   static constexpr uint64_t MASKS[] = {
      0x0000000000000000ULL, // 0 bytes
      0x00000000000000FFULL, // 1 byte
      0x000000000000FFFFULL, // 2 bytes
      0x0000000000FFFFFFULL, // 3 bytes
      0x00000000FFFFFFFFULL, // 4 bytes
      0x000000FFFFFFFFFFULL, // 5 bytes
      0x0000FFFFFFFFFFFFULL, // 6 bytes
      0x00FFFFFFFFFFFFFFULL, // 7 bytes
      0xFFFFFFFFFFFFFFFFULL // 8 bytes
   };
   /// Maximum entries per bucket
   static constexpr size_t MAX_BUCKET_SIZE = 128;

   /// The direct-lookup table for the 2-byte/1-byte tail: entry >= 256 is the id of a 2-byte
   /// symbol, entry < 256 is the (identity) id of the single-byte symbol of the low byte
   std::vector<uint16_t> shortTokenArray;
   /// The dynamic lookup tables for symbols of 3 and 4 bytes (keys left-shifted to the top)
   std::array<TrainMap<uint32_t, uint16_t>, 2> tokenMaps4;
   /// The dynamic lookup tables for symbols of 5 to 8 bytes (keys left-shifted to the top)
   std::array<TrainMap<uint64_t, uint16_t>, 4> tokenMaps8;
   /// The lookup table for long symbols (> 8 bytes): prefix -> [(suffix, suffixLen, id)]
   TrainMap<uint64_t, std::vector<std::tuple<uint64_t, uint8_t, uint16_t>>> longLookupTable;

   /// The static compression table for 3-byte symbols (key = symbol << 8; lossy, so rarely a
   /// 3-byte symbol is dropped and simply never matched) - built by finalize()
   LossyHashMap<uint32_t, uint32_t> tokenMap3;
   /// The static compression table for 4..8 byte symbols, keyed by the first 4 bytes with the
   /// bucket entries sorted longest-first - built by finalize()
   MultiHashMap<uint32_t, ShortTokenEntry> shortTokenMap;
   /// The static compression table for long symbols (9..16 bytes), keyed by the first 8 bytes
   /// with the bucket entries sorted longest-first - built by finalize()
   MultiHashMap<uint64_t, LongTokenEntry> longTokenMap;

   /// Load `len` bytes (<= 8) as a little-endian u64; never reads past bytes + len
   static uint64_t loadPartial(const uint8_t* bytes, const size_t len) {
      uint64_t value = 0;
      std::memcpy(&value, bytes, len);
      return value;
   }
   /// Does what you think it does
   static bool isPrefix(uint64_t text, uint64_t prefix, size_t textLen, size_t prefixLen);

   public:
   /// Constructor: the tail array starts as the identity of the low byte (single-byte symbols)
   Tokenizer() : shortTokenArray(65536) {
      for (size_t i = 0; i < shortTokenArray.size(); ++i) shortTokenArray[i] = static_cast<uint16_t>(i % 256);
   }

   /// Reserve capacity in the lookup tables for up to `tokenCount` tokens
   void reserve(size_t tokenCount) {
      for (auto& map : tokenMaps4) map.reserve(tokenCount / 4);
      for (auto& map : tokenMaps8) map.reserve(tokenCount / 8);
      longLookupTable.reserve(tokenCount / 8);
   }

   /// Inserts a pattern of `length` bytes (must be <= 16) under the given token id; an already
   /// existing pattern keeps its first id. Never reads past data + length.
   /// Returns true if insertion succeeded, false if the bucket was full.
   bool insert(const uint8_t* data, size_t length, uint16_t id);
   /// Finds the longest pattern matching the beginning of the input data (training path, using
   /// the dynamic tables). Returns (token id, match length).
   std::optional<std::pair<uint16_t, size_t>> findLongestMatch(const uint8_t* data, size_t length) const;
   /// Rewrite every stored token id through the remap table (old id -> new id), so that subsequent
   /// findLongestMatch() calls return the new ids directly
   void remapIds(std::span<const uint16_t> remap);
   /// Build the static compression tables from the dynamic ones; call after remapIds().
   /// `firstTrainedId` (the id right past the n-gram block) prioritizes trained symbols over
   /// n-gram trigrams when the lossy 3-byte map has to drop keys.
   void finalize(uint16_t firstTrainedId = 256);

   /// Match the longest symbol starting at `data` against the static compression tables;
   /// `data` must have at least 16 readable bytes (the dense path). finalize() must have been
   /// called. Defined in the header so the hot loop can be inlined.
   std::pair<uint16_t, size_t> match(const uint8_t* data) const {
      uint64_t text;
      std::memcpy(&text, data, sizeof(text));

      // Both directory lookups are issued before either bucket is walked, so their cache misses
      // overlap. The match loop is a serial dependency chain - the next position is only known
      // once this match resolves - so overlapping the two independent misses is a direct win.
      auto [longEntry, longEnd] = longTokenMap.find(text);
      auto [shortEntry, shortEnd] = shortTokenMap.find(static_cast<uint32_t>(text));

      // Long symbols (9..16 bytes), keyed by the first 8 bytes, bucket entries longest-first
      if (longEntry < longEnd) {
         uint64_t postfix;
         std::memcpy(&postfix, data + 8, sizeof(postfix));
         for (; longEntry < longEnd; ++longEntry)
            if (longEntry->prefix == text && (postfix & MASKS[longEntry->length - 8]) == longEntry->postfix)
               return {static_cast<uint16_t>(longEntry->id), longEntry->length};
      }
      // The 4..8 byte bucket, keyed by the first 4 bytes, entries longest-first
      if (!shortTokenMap.empty()) {
         // Fast path: unroll the first 4 entries (the padding past the last bucket never matches)
         for (unsigned i = 0; i < 4; ++i, ++shortEntry)
            if ((text & MASKS[shortEntry->length]) == shortEntry->text) return {static_cast<uint16_t>(shortEntry->id), shortEntry->length};
         for (; shortEntry < shortEnd; ++shortEntry)
            if ((text & MASKS[shortEntry->length]) == shortEntry->text) return {static_cast<uint16_t>(shortEntry->id), shortEntry->length};
      }
      // The 3-byte map (key 0, i.e. three zero bytes, is the map's empty sentinel - skip it)
      if (const auto key = static_cast<uint32_t>(text) << 8; key != 0)
         if (const auto* id = tokenMap3.find(key)) return {static_cast<uint16_t>(*id), 3};
      // The 2-byte/1-byte tail: one array load resolves either a 2-byte symbol or the single byte
      const uint16_t id = shortTokenArray[static_cast<uint16_t>(text)];
      return {id, static_cast<size_t>(1) + (id > 255)};
   }

   /// Match the longest symbol starting at `data` when only `length` (< 16) bytes are readable
   /// (the tail path): the loads are zero-extended past the input's end, so a symbol can only
   /// over-match its own zero bytes, which the decoder's final truncation to the uncompressed
   /// size removes again. Never reads past data + length.
   std::pair<uint16_t, size_t> match(const uint8_t* data, const size_t length) const {
      assert(length > 0 && length < 16);
      const uint64_t text = loadPartial(data, std::min<size_t>(length, 8));

      // Long symbols (9..16 bytes), keyed by the first 8 bytes, bucket entries longest-first
      if (length > 8) {
         auto [entry, end] = longTokenMap.find(text);
         if (entry < end) {
            const uint64_t postfix = loadPartial(data + 8, length - 8);
            for (; entry < end; ++entry)
               if (entry->prefix == text && (postfix & MASKS[entry->length - 8]) == entry->postfix)
                  return {static_cast<uint16_t>(entry->id), entry->length};
         }
      }
      // The 4..8 byte bucket, keyed by the first 4 bytes, entries longest-first
      if (!shortTokenMap.empty()) {
         auto [entry, end] = shortTokenMap.find(static_cast<uint32_t>(text));
         // Fast path: unroll the first 4 entries (the padding past the last bucket never matches)
         for (unsigned i = 0; i < 4; ++i, ++entry)
            if ((text & MASKS[entry->length]) == entry->text) return {static_cast<uint16_t>(entry->id), entry->length};
         for (; entry < end; ++entry)
            if ((text & MASKS[entry->length]) == entry->text) return {static_cast<uint16_t>(entry->id), entry->length};
      }
      // The 3-byte map (key 0, i.e. three zero bytes, is the map's empty sentinel - skip it)
      if (const auto key = static_cast<uint32_t>(text) << 8; key != 0)
         if (const auto* id = tokenMap3.find(key)) return {static_cast<uint16_t>(*id), 3};
      // The 2-byte/1-byte tail: one array load resolves either a 2-byte symbol or the single byte
      const uint16_t id = shortTokenArray[static_cast<uint16_t>(text)];
      return {id, static_cast<size_t>(1) + (id > 255)};
   }
};
//---------------------------------------------------------------------------
}
