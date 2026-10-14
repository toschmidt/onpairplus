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
   Vector<uint128_t> symbols;
   /// The symbol lengths in bytes
   Vector<uint8_t> lengths;
   /// The number of tokens the dictionary defines, set by readDictionary()
   size_t numTokens = 0;

   public:
   /// Constructor: the symbol table allocates from `memory`
   explicit OnPairPlusDecoder(std::pmr::memory_resource* memory = std::pmr::get_default_resource()) : symbols(memory), lengths(memory) {}

   /// Reconstruct the symbol table from the dictionary at the beginning of `input`.
   /// Returns the number of bytes the dictionary used; the token stream starts right after.
   size_t readDictionary(std::span<const std::byte> input);

   /// Whether the stream uses packed 12-bit ids. The width is implied by the dictionary rather
   /// than stored: a dictionary this small is always emitted with 12-bit ids.
   bool uses12BitIds() const noexcept { return numTokens <= MAX_TOKEN_COUNT_12; }

   /// The number of ids a token stream of `tokens.size()` bytes holds. Every two packed ids
   /// occupy three bytes, and a trailing odd id occupies two.
   size_t tokenCount(const std::span<const std::byte> tokens) const noexcept {
      if (!uses12BitIds()) return tokens.size() / sizeof(uint16_t);
      return (tokens.size() / 3) * 2 + (tokens.size() % 3 == 2 ? 1 : 0);
   }

   /// Decode the token stream `tokens` into `output`; readDictionary() must have been called
   /// first, and the caller must size `output` to the uncompressed size plus MAX_SYMBOL_LENGTH
   /// bytes of headroom (the unrolled loop stores 16 bytes per token and advances by the symbol's
   /// true length). Defined in the header so the hot loop can be inlined.
   void decode(const std::span<const std::byte> tokens, const std::span<std::byte> output) const {
      if (uses12BitIds())
         decode12(tokens, output);
      else
         decode16(tokens, output);
   }

   /// Decode an unpacked 16-bit token stream
   void decode16(const std::span<const std::byte> tokens, const std::span<std::byte> output) const {
      auto* writer = output.data();
      const auto* symbolTable = symbols.data();
      const auto* lengthTable = lengths.data();

      const auto* reader = reinterpret_cast<const uint16_t*>(tokens.data());
      const auto* end = reinterpret_cast<const uint16_t*>(tokens.data() + tokens.size());
      for (; reader + 4 <= end; reader += 4) {
         const auto token0 = reader[0];
         unalignedStore(writer, symbolTable[token0]);
         writer += lengthTable[token0];
         const auto token1 = reader[1];
         unalignedStore(writer, symbolTable[token1]);
         writer += lengthTable[token1];
         const auto token2 = reader[2];
         unalignedStore(writer, symbolTable[token2]);
         writer += lengthTable[token2];
         const auto token3 = reader[3];
         unalignedStore(writer, symbolTable[token3]);
         writer += lengthTable[token3];
      }
      for (; reader < end; ++reader) {
         const auto token = reader[0];
         unalignedStore(writer, symbolTable[token]);
         writer += lengthTable[token];
      }
   }

   /// Decode a packed 12-bit token stream. Three 64-bit loads cover 16 ids, so the ids that
   /// straddle a word boundary are stitched from two loads instead of being reloaded.
   void decode12(const std::span<const std::byte> tokens, const std::span<std::byte> output) const {
      auto* writer = output.data();
      const auto* symbolTable = symbols.data();
      const auto* lengthTable = lengths.data();
      const auto* reader = reinterpret_cast<const uint8_t*>(tokens.data());

      const auto emit = [&](const uint64_t id) {
         unalignedStore(writer, symbolTable[id]);
         writer += lengthTable[id];
      };

      size_t remaining = tokenCount(tokens);
      while (remaining >= 16) {
         const uint64_t packed0 = unalignedLoad<uint64_t>(reader);
         const uint64_t packed1 = unalignedLoad<uint64_t>(reader + 8);
         const uint64_t packed2 = unalignedLoad<uint64_t>(reader + 16);
         emit(packed0 & 0xfff);
         emit((packed0 >> 12) & 0xfff);
         emit((packed0 >> 24) & 0xfff);
         emit((packed0 >> 36) & 0xfff);
         emit((packed0 >> 48) & 0xfff);
         emit(((packed0 >> 60) | (packed1 << 4)) & 0xfff);
         emit((packed1 >> 8) & 0xfff);
         emit((packed1 >> 20) & 0xfff);
         emit((packed1 >> 32) & 0xfff);
         emit((packed1 >> 44) & 0xfff);
         emit(((packed1 >> 56) | (packed2 << 8)) & 0xfff);
         emit((packed2 >> 4) & 0xfff);
         emit((packed2 >> 16) & 0xfff);
         emit((packed2 >> 28) & 0xfff);
         emit((packed2 >> 40) & 0xfff);
         emit((packed2 >> 52) & 0xfff);
         reader += 24;
         remaining -= 16;
      }
      // The tail (at most 15 ids) reads byte by byte, so the loop never loads past the payload
      // and the stream needs no trailing padding
      while (remaining >= 2) {
         const uint32_t packed = static_cast<uint32_t>(reader[0]) | (static_cast<uint32_t>(reader[1]) << 8) | (static_cast<uint32_t>(reader[2]) << 16);
         emit(packed & 0xfff);
         emit(packed >> 12);
         reader += 3;
         remaining -= 2;
      }
      if (remaining) emit((static_cast<uint32_t>(reader[0]) | (static_cast<uint32_t>(reader[1]) << 8)) & 0xfff);
   }
};
//---------------------------------------------------------------------------
}
