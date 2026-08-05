#pragma once
//---------------------------------------------------------------------------
#include <span>
#include <stdexcept>
#include <vector>
#include <zstd.h>
//---------------------------------------------------------------------------
// Zstandard adapter (level 3, the default)
//
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace bench::zstd {
//---------------------------------------------------------------------------
inline constexpr const char* name = "zstd";
inline constexpr int LEVEL = 3;
//---------------------------------------------------------------------------
inline void compress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   output.resize(ZSTD_compressBound(input.size()));
   const size_t written = ZSTD_compress(output.data(), output.size(), input.data(), input.size(), LEVEL);
   if (ZSTD_isError(written)) throw std::runtime_error("zstd: compression failed");
   output.resize(written);
}
//---------------------------------------------------------------------------
inline void decompress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   const auto originalSize = ZSTD_getFrameContentSize(input.data(), input.size());
   if (originalSize == ZSTD_CONTENTSIZE_UNKNOWN || originalSize == ZSTD_CONTENTSIZE_ERROR)
      throw std::runtime_error("zstd: unknown content size");
   output.resize(originalSize);
   const size_t written = ZSTD_decompress(output.data(), output.size(), input.data(), input.size());
   if (ZSTD_isError(written) || written != originalSize) throw std::runtime_error("zstd: decompression failed");
}
//---------------------------------------------------------------------------
}
