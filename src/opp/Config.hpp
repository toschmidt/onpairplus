#pragma once
//---------------------------------------------------------------------------
#include "util/Infra.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//
// Shared constants, the n-gram level descriptor, and the compressed stream layout:
// [uncompressedSize u64][dictionary][packed 12-bit or 16-bit token stream]
// where the dictionary is [(MAX_NGRAM_ORDER-1) char bitmaps][16-entry length histogram]
// [short raw bytes][long recipes]. The dictionary is self-delimiting: the length histogram
// determines the size of the short and recipe blocks, so no offsets need to be stored.
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
/// The maximum length of a dictionary symbol in bytes
constexpr unsigned MAX_SYMBOL_LENGTH = 16;
/// The maximum number of tokens (16-bit token ids)
constexpr unsigned MAX_TOKEN_COUNT = 65536;
/// The maximum number of dictionary tokens still addressable with 12-bit ids. A dictionary of at
/// most this many tokens implies a packed 12-bit stream, so the width never has to be stored.
constexpr unsigned MAX_TOKEN_COUNT_12 = 4096;
/// The maximum symbol length (in bytes) stored as raw bytes in the hybrid dictionary format;
/// longer symbols are stored as the 4-byte merge recipe instead (min(length, 4) per symbol).
constexpr uint32_t HYBRID_SHORT_MAX = 4;
/// The largest n-gram order seeded into the leftover token space (digrams=2, trigrams=3, fourgrams=4)
constexpr unsigned MAX_NGRAM_ORDER = 4;
/// The number of entries in the serialized length histogram
constexpr unsigned MAX_LENGTHS = 16;
/// The size of one serialized n-gram character bitmap
constexpr size_t BITMAP_BYTES = 32;
/// The size of all n-gram bitmaps at the start of the dictionary region
constexpr size_t HEADER_BYTES = (MAX_NGRAM_ORDER - 1) * BITMAP_BYTES;
/// A tight upper bound for the serialized dictionary size: the n-gram bitmaps, the length
/// histogram, and at most 4 bytes per trained symbol (raw bytes for length <= HYBRID_SHORT_MAX,
/// a 2x uint16 merge recipe otherwise)
constexpr size_t MAX_DICTIONARY_SIZE = HEADER_BYTES + MAX_LENGTHS * sizeof(uint16_t) + MAX_TOKEN_COUNT * 2 * sizeof(uint16_t);
/// The chunk size for dictionary training: the input is split into chunks of this size which are
/// visited in a seeded-shuffled order, so the sample the dictionary is trained on (until the table
/// cutoff) is drawn evenly from the whole input instead of just its beginning
constexpr size_t TRAIN_CHUNK_SIZE = 1 << 16;
/// The maximum sample size for dictionary training: enough to fill the 65536-entry token table
/// even at the highest merge threshold, while bounding the training cost on huge inputs
constexpr size_t MAX_TRAIN_SIZE = 128 << 20;
/// The maximum number of sampled training chunks
constexpr size_t MAX_TRAIN_CHUNKS = MAX_TRAIN_SIZE / TRAIN_CHUNK_SIZE;
/// The number of input bytes tokenized with each candidate dictionary to choose the token id
/// width. The decision is a ratio of two token counts and settles well before this much data:
/// measured over dbtext, bitext and enwik9, every cap from here upwards picks the same width on
/// all but a handful of files near the crossover, while the cost grows linearly with the prefix.
constexpr size_t WIDTH_EVAL_SIZE = 256 << 10;
//---------------------------------------------------------------------------
/// One seeded n-gram level: all K^n combinations of the top-K characters (n bytes each). `base` is
/// the first token id of the level; `chars` are the top-K characters in ascending byte order (the
/// canonical enumeration order) and `idx` maps a byte to its position in `chars` (-1 if not present).
struct NgramLevel {
   unsigned n = 0;
   size_t K = 0;
   size_t base = 0;
   std::array<uint8_t, BITMAP_BYTES> bitmap{};
   std::array<uint8_t, 256> chars{};
   std::array<int16_t, 256> idx{};
   size_t count() const {
      size_t c = 1;
      for (unsigned t = 0; t < n; ++t) c *= K;
      return c;
   }
};
//---------------------------------------------------------------------------
/// Enumerate the little-endian byte values of all K^n n-grams of a level in canonical id order
/// (byte t = chars[digit t], the last character varies fastest), matching ngramIdOf. Plain nested
/// loops per order: the inner loops only patch the highest byte, no divisions needed.
template <typename Fn>
void forEachNgram(const NgramLevel& L, Fn&& fn) {
   static_assert(MAX_NGRAM_ORDER == 4, "forEachNgram covers n = 2..4");
   const auto* chars = L.chars.data();
   const size_t K = L.K;
   switch (L.n) {
      case 2:
         for (size_t a = 0; a < K; ++a) {
            const uint64_t va = chars[a];
            for (size_t b = 0; b < K; ++b) fn(va | static_cast<uint64_t>(chars[b]) << 8);
         }
         break;
      case 3:
         for (size_t a = 0; a < K; ++a) {
            const uint64_t va = chars[a];
            for (size_t b = 0; b < K; ++b) {
               const uint64_t vb = va | static_cast<uint64_t>(chars[b]) << 8;
               for (size_t c = 0; c < K; ++c) fn(vb | static_cast<uint64_t>(chars[c]) << 16);
            }
         }
         break;
      case 4:
         for (size_t a = 0; a < K; ++a) {
            const uint64_t va = chars[a];
            for (size_t b = 0; b < K; ++b) {
               const uint64_t vb = va | static_cast<uint64_t>(chars[b]) << 8;
               for (size_t c = 0; c < K; ++c) {
                  const uint64_t vc = vb | static_cast<uint64_t>(chars[c]) << 16;
                  for (size_t d = 0; d < K; ++d) fn(vc | static_cast<uint64_t>(chars[d]) << 24);
               }
            }
         }
         break;
      default: break;
   }
}
//---------------------------------------------------------------------------
/// If the `len`-byte symbol is exactly one of this level's n-grams (right n, all bytes in the set),
/// return its final id; otherwise -1.
inline int64_t ngramIdOf(const NgramLevel& L, const uint8_t* bytes, const size_t len) {
   if (L.K == 0 || len != L.n) return -1;
   size_t id = 0;
   for (size_t t = 0; t < len; ++t) {
      const int p = L.idx[bytes[t]];
      if (p < 0) return -1;
      id = id * L.K + static_cast<size_t>(p);
   }
   return static_cast<int64_t>(L.base + id);
}
//---------------------------------------------------------------------------
}
