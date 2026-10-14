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
   /// The memory resource every table allocates from
   std::pmr::memory_resource* memory;
   /// The tokenizer used for training and compression
   Tokenizer tokenizer;
   /// The input, set by trainDictionary(); must remain valid until compress() has been called
   std::span<const uint8_t> input;

   /// The seeded n-gram levels
   Vector<NgramLevel> levels;
   /// The finalized length histogram of the dictionary (256 singles + kept trained symbols)
   std::array<uint16_t, MAX_LENGTHS> lengthHistogram{};
   /// The raw bytes of the kept trained symbols of length 2..HYBRID_SHORT_MAX, in final id order
   Vector<uint8_t> shortSymbols;
   /// The merge recipes (two parent final-ids) of the longer kept trained symbols, in final id order
   Vector<uint16_t> recipes;
   /// The physical id width chosen for the token stream (12 or 16 bits)
   uint8_t tokenBitsValue = 16;
   /// The id counts the two candidates produced on the evaluation prefix (0 if not evaluated)
   uint64_t tokenCount12Value = 0;
   uint64_t tokenCount16Value = 0;
   /// Whether the full-table candidate was built and counted at all
   bool comparedFullTableValue = false;

   /// One finalized dictionary: its tables, the blocks serializeDictionary() writes, and the
   /// number of tokens it defines (which implies the stream's id width)
   struct Candidate {
      Tokenizer tokenizer;
      Vector<NgramLevel> levels;
      std::array<uint16_t, MAX_LENGTHS> lengthHistogram{};
      Vector<uint8_t> shortSymbols;
      Vector<uint16_t> recipes;
      size_t dictionaryTokens = 0;

      /// Constructor
      explicit Candidate(std::pmr::memory_resource* memory) : tokenizer(memory), levels(memory), shortSymbols(memory), recipes(memory) {}
   };

   public:
   /// Constructor: every table of the encoder allocates from `memory`
   explicit OnPairPlusEncoder(std::pmr::memory_resource* memory = std::pmr::get_default_resource()) : memory(memory), tokenizer(memory), levels(memory), shortSymbols(memory), recipes(memory) {}

   /// Train a dictionary on the input (streaming merge pass + n-gram seeding of the leftover
   /// token-id space), initialize the tokenizer with final token ids, and build the finalized
   /// dictionary blocks
   void trainDictionary(std::span<const std::byte> input);
   /// Serialize the finalized dictionary into `out` ([char bitmaps][length histogram][short raw
   /// bytes][long recipes]). Returns the serialized size.
   size_t serializeDictionary(std::span<std::byte> out) const;
   /// Tokenize the input into `out` as a stream of final token ids, packed to the width chosen by
   /// trainDictionary(). Returns the number of bytes written.
   size_t compress(std::span<std::byte> out) const;

   /// The physical id width chosen for the token stream (12 or 16 bits)
   uint8_t tokenBits() const { return tokenBitsValue; }
   /// The id count the 12-bit candidate produced on the evaluation prefix
   uint64_t tokenCount12() const { return tokenCount12Value; }
   /// The id count the full-table candidate produced, or 0 if it was not evaluated
   uint64_t tokenCount16() const { return tokenCount16Value; }
   /// Whether the full-table candidate was built and counted
   bool comparedFullTable() const { return comparedFullTableValue; }
};
//---------------------------------------------------------------------------
}
