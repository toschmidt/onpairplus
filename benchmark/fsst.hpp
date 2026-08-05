#pragma once
//---------------------------------------------------------------------------
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>
#include <fsst.h>
//---------------------------------------------------------------------------
// FSST adapter (https://github.com/cwida/fsst). The whole input is compressed
// as a single string with one symbol table.
//
// Compressed layout: [u64 originalSize][u32 headerLen][exported symbol table][compressed bytes]
//
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace bench::fsst {
//---------------------------------------------------------------------------
inline constexpr const char* name = "fsst";
//---------------------------------------------------------------------------
inline void compress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   const uint64_t originalSize = input.size();
   size_t lenIn = input.size();
   const auto* strIn = reinterpret_cast<const unsigned char*>(input.data());

   const std::unique_ptr<fsst_encoder_t, decltype(&fsst_destroy)> encoder(
      fsst_create(1, &lenIn, &strIn, 0), &fsst_destroy);
   if (!encoder) throw std::runtime_error("fsst: fsst_create failed");

   unsigned char tableBuffer[FSST_MAXHEADER];
   const uint32_t headerLen = fsst_export(encoder.get(), tableBuffer);

   // Worst case: every byte becomes an escape sequence (2 bytes)
   const size_t metaLen = sizeof(originalSize) + sizeof(headerLen) + headerLen;
   output.resize(metaLen + 2 * input.size() + 64);
   std::memcpy(output.data(), &originalSize, sizeof(originalSize));
   std::memcpy(output.data() + sizeof(originalSize), &headerLen, sizeof(headerLen));
   std::memcpy(output.data() + sizeof(originalSize) + sizeof(headerLen), tableBuffer, headerLen);

   size_t lenOut = 0;
   unsigned char* strOut = nullptr;
   const size_t compressed = fsst_compress(
      encoder.get(), 1, &lenIn, &strIn, output.size() - metaLen,
      reinterpret_cast<unsigned char*>(output.data() + metaLen), &lenOut, &strOut);
   if (compressed != 1) throw std::runtime_error("fsst: compression failed");
   output.resize(metaLen + lenOut);
}
//---------------------------------------------------------------------------
inline void decompress(std::span<const std::byte> input, std::vector<std::byte>& output) {
   uint64_t originalSize;
   uint32_t headerLen;
   std::memcpy(&originalSize, input.data(), sizeof(originalSize));
   std::memcpy(&headerLen, input.data() + sizeof(originalSize), sizeof(headerLen));
   const size_t metaLen = sizeof(originalSize) + sizeof(headerLen) + headerLen;

   fsst_decoder_t decoder;
   std::vector<unsigned char> tableBuffer(
      reinterpret_cast<const unsigned char*>(input.data() + sizeof(originalSize) + sizeof(headerLen)),
      reinterpret_cast<const unsigned char*>(input.data() + metaLen));
   fsst_import(&decoder, tableBuffer.data());

   // The decoder may store a few bytes past the last symbol, so decode with headroom
   output.resize(originalSize + 32);
   const size_t written = fsst_decompress(
      &decoder, input.size() - metaLen, reinterpret_cast<const unsigned char*>(input.data() + metaLen),
      output.size(), reinterpret_cast<unsigned char*>(output.data()));
   if (written != originalSize) throw std::runtime_error("fsst: decompression failed");
   output.resize(originalSize);
}
//---------------------------------------------------------------------------
}
