#pragma once
//---------------------------------------------------------------------------
#include "opp/Config.hpp"
#include "opp/Tokenizer.hpp"
#include <array>
#include <span>
#include <vector>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
/// The OnPair+ encoder: trainDictionary() trains the dictionary in a single streaming pass, seeds
/// the leftover token-id space with n-grams of the most frequent characters, and finalizes the
/// token ids (length-sort + n-gram dedup) both in the tokenizer and in the stored dictionary, so
/// no id remapping survives training; serializeDictionary() writes the finalized hybrid format;
/// compress() tokenizes the input directly into final ids.
class OnPairPlusEncoder {
   /// The tokenizer used for training and compression
   Tokenizer tokenizer;
   /// The input, set by trainDictionary(); must remain valid until compress() has been called
   std::span<const uint8_t> input;

   /// The seeded n-gram levels
   std::vector<NgramLevel> levels;
   /// The finalized length histogram of the dictionary (256 singles + kept trained symbols)
   std::array<uint16_t, MAX_LENGTHS> lengthHistogram{};
   /// The raw bytes of the kept trained symbols of length 2..HYBRID_SHORT_MAX, in final id order
   std::vector<uint8_t> shortSymbols;
   /// The merge recipes (two parent final-ids) of the longer kept trained symbols, in final id order
   std::vector<uint16_t> recipes;

   public:
   /// Train a dictionary on the input (streaming merge pass + n-gram seeding of the leftover
   /// token-id space), initialize the tokenizer with final token ids, and build the finalized
   /// dictionary blocks
   void trainDictionary(std::span<const std::byte> input);
   /// Serialize the finalized dictionary into `out` ([char bitmaps][length histogram][short raw
   /// bytes][long recipes]). Returns the serialized size.
   size_t serializeDictionary(std::span<std::byte> out) const;
   /// Tokenize the input into `out` as a stream of final token ids. Returns the number of bytes written.
   size_t compress(std::span<std::byte> out) const;
};
//---------------------------------------------------------------------------
}
