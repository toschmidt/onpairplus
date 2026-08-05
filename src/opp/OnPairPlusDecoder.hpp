#pragma once
//---------------------------------------------------------------------------
#include "opp/Config.hpp"
#include "util/Infra.hpp"
#include <span>
#include <vector>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
/// The OnPair+ decoder: readDictionary() reconstructs the symbol table (singles, n-gram levels
/// from the bitmaps, short raw symbols, merge recipes); decode() decodes the token stream.
class OnPairPlusDecoder {
   /// The materialized symbols (16 bytes each; writes may spill into the next entry, hence +1
   /// sentinel). Sized by readDictionary() to the exact token count of the stream's dictionary.
   std::vector<uint128_t> symbols;
   /// The symbol lengths in bytes
   std::vector<uint8_t> lengths;

   public:
   /// Reconstruct the symbol table from the dictionary at the beginning of `input`.
   /// Returns the number of bytes the dictionary used; the token stream starts right after.
   size_t readDictionary(std::span<const std::byte> input);

   /// Decode the token stream `tokens` into `output`; readDictionary() must have been called
   /// first, and the caller must size `output` to the uncompressed size plus MAX_SYMBOL_LENGTH
   /// bytes of headroom (the unrolled loop stores 16 bytes per token and advances by the symbol's
   /// true length). Defined in the header so the hot loop can be inlined.
   void decode(const std::span<const std::byte> tokens, const std::span<std::byte> output) const {
      auto* writer = output.data();

      const auto* reader = reinterpret_cast<const uint16_t*>(tokens.data());
      const auto* end = reinterpret_cast<const uint16_t*>(tokens.data() + tokens.size());
      for (; reader + 4 <= end; reader += 4) {
         const auto token0 = reader[0];
         unalignedStore(writer, symbols[token0]);
         writer += lengths[token0];
         const auto token1 = reader[1];
         unalignedStore(writer, symbols[token1]);
         writer += lengths[token1];
         const auto token2 = reader[2];
         unalignedStore(writer, symbols[token2]);
         writer += lengths[token2];
         const auto token3 = reader[3];
         unalignedStore(writer, symbols[token3]);
         writer += lengths[token3];
      }
      for (; reader < end; ++reader) {
         const auto token = reader[0];
         unalignedStore(writer, symbols[token]);
         writer += lengths[token];
      }
   }
};
//---------------------------------------------------------------------------
}
