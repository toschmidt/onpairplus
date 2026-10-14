#pragma once
//---------------------------------------------------------------------------
#include "util/Infra.hpp"
#include <cstddef>
#include <memory_resource>
#include <span>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//
// OnPair+ trains a BPE-style dictionary of up to 65536 symbols (max 16 bytes each) in a single
// streaming pass over the input, then fills the leftover token-id space with n-grams of the most
// frequent characters (n = 2..4) so the payload never expands past a 2:1 floor. The dictionary is
// serialized in a hybrid format: short symbols (2..4 bytes) as raw bytes, longer ones as a merge
// recipe of two parent token ids.
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
/// Compress `input` into `output` (replacing its previous contents); every allocation of the encoder
/// comes from the memory resource of `output`
void encode(std::span<const std::byte> input, Vector<std::byte>& output);
/// Decompress `input` (as produced by encode) into `output` (replacing its previous contents); the
/// symbol table comes from the memory resource of `output`
void decode(std::span<const std::byte> input, Vector<std::byte>& output);
/// The uncompressed size of `input` (as produced by encode)
size_t decodedSize(std::span<const std::byte> input);
/// Decompress `input` into `output`, which must hold decodedSize(input) + 16 bytes (the decoder
/// stores 16 bytes per symbol); the bytes past the decoded size are overwritten. Returns the decoded size
/// The symbol table comes from `memory`.
size_t decodeInto(std::span<const std::byte> input, std::span<std::byte> output, std::pmr::memory_resource* memory = std::pmr::get_default_resource());
//---------------------------------------------------------------------------
}
