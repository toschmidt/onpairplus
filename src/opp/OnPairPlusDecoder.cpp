#include "opp/OnPairPlusDecoder.hpp"
#include <bit>
#include <cstring>
#include <stdexcept>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
size_t OnPairPlusDecoder::readDictionary(const std::span<const std::byte> input)
// Reconstruct the symbol table from the dictionary at the beginning of the input. Layout: 256
// singles, then for n = 2..MAX_NGRAM_ORDER all K_n^n n-grams of the top-K_n characters rebuilt
// from that level's 256-bit bitmap, then the kept trained symbols, length-sorted (n-grams that
// duplicated a trained symbol were dropped at encode). Trained recipes already store final ids,
// so no id shift is needed. Returns the number of bytes the dictionary used.
{
   constexpr size_t histSize = MAX_LENGTHS * sizeof(uint16_t);
   if (input.size() < HEADER_BYTES + histSize) throw std::runtime_error("opp::decode: input too small");

   // Dictionary layout: [(MAX_NGRAM_ORDER-1) char bitmaps][16-entry length histogram][short raw
   // bytes][long recipes]. The histogram/short/recipes are the deduplicated trained dictionary.
   const auto* header = reinterpret_cast<const uint8_t*>(input.data());
   const auto* counts = reinterpret_cast<const uint16_t*>(input.data() + HEADER_BYTES);

   // Size the tables to the exact token count of this dictionary: 256 singles, each level's K^n
   // n-grams (K = the popcount of its bitmap), and the kept trained symbols from the length
   // histogram, plus one sentinel entry (recipe construction may spill 16 bytes into the next entry)
   size_t totalTokens = 256;
   for (unsigned n = 2; n <= MAX_NGRAM_ORDER; ++n) {
      size_t K = 0;
      const uint8_t* bm = header + (n - 2) * BITMAP_BYTES;
      for (size_t b = 0; b < BITMAP_BYTES; ++b) K += static_cast<size_t>(std::popcount(bm[b]));
      size_t count = K;
      for (unsigned t = 1; t < n; ++t) count *= K;
      if (K) totalTokens += count;
   }
   for (unsigned l = 2; l <= MAX_LENGTHS; ++l) totalTokens += counts[l - 1];
   numTokens = totalTokens;
   symbols.resize(totalTokens + 1);
   lengths.resize(totalTokens + 1);

   // 256 base singles (value == id), then each n-gram level's K_n^n combinations
   size_t newId = 0;
   for (; newId < 256; ++newId) {
      symbols[newId] = static_cast<uint128_t>(newId);
      lengths[newId] = 1;
   }
   for (unsigned n = 2; n <= MAX_NGRAM_ORDER; ++n) {
      NgramLevel L;
      L.n = n;
      const uint8_t* bm = header + (n - 2) * BITMAP_BYTES;
      for (unsigned b = 0; b < 256; ++b)
         if (bm[b >> 3] & (1u << (b & 7))) L.chars[L.K++] = static_cast<uint8_t>(b);
      if (L.K == 0) continue;
      forEachNgram(L, [&](uint64_t value) {
         symbols[newId] = static_cast<uint128_t>(value);
         lengths[newId] = static_cast<uint8_t>(n);
         ++newId;
      });
   }

   // Kept trained symbols follow (length-sorted). Skip the histogram's length-1 count (the 256
   // singles, already added). Recipe parent ids are already final, so index symbols[] directly.
   size_t shortBytes = 0;
   for (unsigned l = 2; l <= HYBRID_SHORT_MAX; ++l) shortBytes += static_cast<size_t>(counts[l - 1]) * l;
   const auto* shortReader = reinterpret_cast<const uint8_t*>(input.data() + HEADER_BYTES + histSize);
   size_t byteOffset = 0;
   for (unsigned l = 2; l <= HYBRID_SHORT_MAX; ++l) {
      for (unsigned c = 0; c < counts[l - 1]; ++c, ++newId) {
         uint128_t symbol = 0;
         std::memcpy(&symbol, shortReader + byteOffset, l);
         symbols[newId] = symbol;
         lengths[newId] = static_cast<uint8_t>(l);
         byteOffset += l;
      }
   }
   const auto* recipe = reinterpret_cast<const uint16_t*>(shortReader + shortBytes);
   size_t recipeCount = 0;
   for (unsigned l = HYBRID_SHORT_MAX + 1; l <= MAX_LENGTHS; ++l) {
      recipeCount += counts[l - 1];
      for (unsigned c = 0; c < counts[l - 1]; ++c, ++newId, recipe += 2) {
         const uint16_t token0 = recipe[0];
         const uint16_t token1 = recipe[1];
         const auto len0 = lengths[token0];
         const auto len1 = lengths[token1];
         auto* symbolWriter = reinterpret_cast<uint8_t*>(&symbols[newId]);
         std::memcpy(symbolWriter, &symbols[token0], MAX_SYMBOL_LENGTH);
         symbolWriter += len0;
         std::memcpy(symbolWriter, &symbols[token1], MAX_SYMBOL_LENGTH);
         lengths[newId] = len0 + len1;
      }
   }

   return HEADER_BYTES + histSize + shortBytes + recipeCount * 2 * sizeof(uint16_t);
}
//---------------------------------------------------------------------------
}
