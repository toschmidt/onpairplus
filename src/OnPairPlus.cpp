#include "OnPairPlus.hpp"
#include "opp/OnPairPlusDecoder.hpp"
#include "opp/OnPairPlusEncoder.hpp"
#include "util/Infra.hpp"
#include <stdexcept>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
void encode(const std::span<const std::byte> input, std::vector<std::byte>& output)
// Compress `input` into `output` (replacing its previous contents).
// Stream layout: [uncompressedSize u64][dictionary][u16 token stream]
{
   const uint64_t totalSize = input.size();

   OnPairPlusEncoder encoder;
   encoder.trainDictionary(input);

   // Ensure enough space: size prefix + dictionary + worst-case token stream (one token per byte)
   output.resize(sizeof(totalSize) + MAX_DICTIONARY_SIZE + 2 * totalSize);

   unalignedStore<uint64_t>(output.data(), totalSize);
   const size_t dictSize = encoder.serializeDictionary({output.data() + sizeof(totalSize), MAX_DICTIONARY_SIZE});
   const size_t payloadSize = encoder.compress({output.data() + sizeof(totalSize) + dictSize, output.size() - sizeof(totalSize) - dictSize});

   // Truncate the output to actual size
   output.resize(sizeof(totalSize) + dictSize + payloadSize);
}
//---------------------------------------------------------------------------
void decode(const std::span<const std::byte> input, std::vector<std::byte>& output)
// Decompress `input` (as produced by encode) into `output` (replacing its previous contents)
{
   if (input.size() < sizeof(uint64_t)) throw std::runtime_error("opp::decode: input too small");
   const auto totalSize = unalignedLoad<uint64_t>(input.data());

   OnPairPlusDecoder decoder;
   const size_t dictSize = decoder.readDictionary(input.subspan(sizeof(uint64_t)));

   // Ensure the output is large enough: the decode loop stores 16 bytes per token, so it needs
   // MAX_SYMBOL_LENGTH bytes of headroom that are truncated away afterwards
   output.resize(totalSize + MAX_SYMBOL_LENGTH);
   decoder.decode(input.subspan(sizeof(uint64_t) + dictSize), output);
   output.resize(totalSize);
}
//---------------------------------------------------------------------------
}
