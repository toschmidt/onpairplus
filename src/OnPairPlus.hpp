#pragma once
//---------------------------------------------------------------------------
#include <cstddef>
#include <span>
#include <vector>
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
/// Compress `input` into `output` (replacing its previous contents)
void encode(std::span<const std::byte> input, std::vector<std::byte>& output);
/// Decompress `input` (as produced by encode) into `output` (replacing its previous contents)
void decode(std::span<const std::byte> input, std::vector<std::byte>& output);
//---------------------------------------------------------------------------
}
