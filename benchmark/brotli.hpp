#pragma once
//---------------------------------------------------------------------------
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>
#include <brotli/decode.h>
#include <brotli/encode.h>
//---------------------------------------------------------------------------
// Brotli adapter (quality 5, default window). The stream does not carry the
// original size, so it is prepended as a uint64.
//
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace bench::brotli {
//---------------------------------------------------------------------------
inline constexpr const char* name = "brotli";
inline constexpr int QUALITY = 5;
//---------------------------------------------------------------------------
inline void compress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   const uint64_t originalSize = input.size();
   size_t written = BrotliEncoderMaxCompressedSize(input.size());
   output.resize(sizeof(originalSize) + written);
   std::memcpy(output.data(), &originalSize, sizeof(originalSize));
   const auto ok = BrotliEncoderCompress(
      QUALITY, BROTLI_DEFAULT_WINDOW, BROTLI_DEFAULT_MODE, input.size(),
      reinterpret_cast<const uint8_t*>(input.data()), &written,
      reinterpret_cast<uint8_t*>(output.data() + sizeof(originalSize)));
   if (!ok) throw std::runtime_error("brotli: compression failed");
   output.resize(sizeof(originalSize) + written);
}
//---------------------------------------------------------------------------
inline void decompress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   uint64_t originalSize;
   std::memcpy(&originalSize, input.data(), sizeof(originalSize));
   output.resize(originalSize);
   size_t written = originalSize;
   const auto result = BrotliDecoderDecompress(
      input.size() - sizeof(originalSize), reinterpret_cast<const uint8_t*>(input.data() + sizeof(originalSize)),
      &written, reinterpret_cast<uint8_t*>(output.data()));
   if (result != BROTLI_DECODER_RESULT_SUCCESS || written != originalSize) throw std::runtime_error("brotli: decompression failed");
}
//---------------------------------------------------------------------------
}
