#pragma once
//---------------------------------------------------------------------------
#include <algorithm>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>
#include <lz4.h>
//---------------------------------------------------------------------------
// LZ4 adapter (default/fast level). LZ4 blocks do not carry the original
// size, so it is prepended as a uint64. Inputs beyond LZ4's single-call
// limit are compressed as a sequence of independent 1 GiB blocks, each
// prefixed with its compressed size as a uint32.
//
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace bench::lz4 {
//---------------------------------------------------------------------------
inline constexpr const char* name = "lz4";
/// Block size for chunked compression (LZ4_MAX_INPUT_SIZE is ~2.1 GB)
inline constexpr size_t BLOCK_SIZE = 1ull << 30;
//---------------------------------------------------------------------------
inline void compress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   const uint64_t originalSize = input.size();
   const size_t blockCount = originalSize == 0 ? 1 : (originalSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
   output.resize(sizeof(originalSize) + blockCount * (sizeof(uint32_t) + LZ4_compressBound(BLOCK_SIZE)));
   std::memcpy(output.data(), &originalSize, sizeof(originalSize));

   size_t outPos = sizeof(originalSize);
   for (size_t offset = 0; offset < originalSize || offset == 0; offset += BLOCK_SIZE) {
      const auto blockSize = static_cast<int>(std::min(BLOCK_SIZE, input.size() - offset));
      const int written = LZ4_compress_default(
         reinterpret_cast<const char*>(input.data() + offset),
         reinterpret_cast<char*>(output.data() + outPos + sizeof(uint32_t)),
         blockSize, LZ4_compressBound(blockSize));
      if (written <= 0 && blockSize > 0) throw std::runtime_error("lz4: compression failed");
      const auto compressedSize = static_cast<uint32_t>(written);
      std::memcpy(output.data() + outPos, &compressedSize, sizeof(compressedSize));
      outPos += sizeof(compressedSize) + compressedSize;
      if (originalSize == 0) break;
   }
   output.resize(outPos);
}
//---------------------------------------------------------------------------
inline void decompress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   uint64_t originalSize;
   std::memcpy(&originalSize, input.data(), sizeof(originalSize));
   output.resize(originalSize);

   size_t inPos = sizeof(originalSize);
   for (size_t offset = 0; offset < originalSize; offset += BLOCK_SIZE) {
      const auto blockSize = static_cast<int>(std::min<uint64_t>(BLOCK_SIZE, originalSize - offset));
      uint32_t compressedSize;
      std::memcpy(&compressedSize, input.data() + inPos, sizeof(compressedSize));
      inPos += sizeof(compressedSize);
      const int written = LZ4_decompress_safe(
         reinterpret_cast<const char*>(input.data() + inPos), reinterpret_cast<char*>(output.data() + offset),
         static_cast<int>(compressedSize), blockSize);
      if (written != blockSize) throw std::runtime_error("lz4: decompression failed");
      inPos += compressedSize;
   }
}
//---------------------------------------------------------------------------
}
