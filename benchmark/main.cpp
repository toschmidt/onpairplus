#include "brotli.hpp"
#include "fsst.hpp"
#include "lz4.hpp"
#include "opp.hpp"
#include "tokenizer.hpp"
#include "zstd.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
//---------------------------------------------------------------------------
// Benchmark driver: reads a file into memory, runs every compressor in-process for the given
// number of repetitions (keeping the best time), verifies each roundtrip, and prints one CSV
// line per compressor: name,originalSize,compressedSize,bestCompressSeconds,bestDecompressSeconds
//
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
struct Compressor {
   const char* name;
   std::function<void(std::span<const std::byte>, std::vector<std::byte>&)> compress;
   std::function<void(std::span<const std::byte>, std::vector<std::byte>&)> decompress;
};
//---------------------------------------------------------------------------
/// Chains two compressors: compress with `first`, then compress the result with `second`
/// (decompression runs in reverse). The intermediate buffer is part of the measured time.
Compressor chain(const char* name, const Compressor& first, const Compressor& second) {
   return {
      name,
      [&first, &second](std::span<const std::byte> input, std::vector<std::byte>& output) {
         std::vector<std::byte> intermediate;
         first.compress(input, intermediate);
         second.compress(intermediate, output);
      },
      [&first, &second](std::span<const std::byte> input, std::vector<std::byte>& output) {
         std::vector<std::byte> intermediate;
         second.decompress(input, intermediate);
         first.decompress(intermediate, output);
      },
   };
}
//---------------------------------------------------------------------------
const Compressor LZ4 = {bench::lz4::name, bench::lz4::compress, bench::lz4::decompress};
const Compressor ZSTD = {bench::zstd::name, bench::zstd::compress, bench::zstd::decompress};
const Compressor OPP = {bench::opp::name, bench::opp::compress, bench::opp::decompress};
//---------------------------------------------------------------------------
const Compressor COMPRESSORS[] = {
   LZ4,
   ZSTD,
   {bench::brotli::name, bench::brotli::compress, bench::brotli::decompress},
   {bench::fsst::name, bench::fsst::compress, bench::fsst::decompress},
   {bench::tokenizer::name, bench::tokenizer::compress, bench::tokenizer::decompress},
   OPP,
   chain("opp+lz4", OPP, LZ4),
   chain("opp+zstd", OPP, ZSTD),
};
//---------------------------------------------------------------------------
std::vector<std::byte> readFile(const std::string& path) {
   std::ifstream file(path, std::ios::binary | std::ios::ate);
   if (!file) throw std::runtime_error("cannot open input file: " + path);
   const auto size = file.tellg();
   file.seekg(0);
   std::vector<std::byte> data(static_cast<size_t>(size));
   file.read(reinterpret_cast<char*>(data.data()), size);
   if (!file) throw std::runtime_error("cannot read input file: " + path);
   return data;
}
//---------------------------------------------------------------------------
template <typename Fn>
double bestOf(const unsigned repeats, Fn&& fn) {
   double best = std::numeric_limits<double>::infinity();
   for (unsigned r = 0; r < repeats; ++r) {
      const auto begin = std::chrono::steady_clock::now();
      fn();
      const auto end = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double>(end - begin).count());
   }
   return best;
}
//---------------------------------------------------------------------------
}
//---------------------------------------------------------------------------
int main(int argc, char** argv) {
   if (argc < 2 || argc > 3) {
      std::cerr << "Usage: " << argv[0] << " <file> [repeats=3]\n";
      return 1;
   }
   const unsigned repeats = argc == 3 ? std::max(1, std::atoi(argv[2])) : 3;

   try {
      const auto input = readFile(argv[1]);

      for (const auto& compressor : COMPRESSORS) {
         std::vector<std::byte> compressed, restored;
         const double compressSeconds = bestOf(repeats, [&] { compressor.compress(input, compressed); });
         const double decompressSeconds = bestOf(repeats, [&] { compressor.decompress(compressed, restored); });
         if (restored != input) throw std::runtime_error(std::string(compressor.name) + ": roundtrip mismatch");
         std::printf("%s,%zu,%zu,%.9f,%.9f\n", compressor.name, input.size(), compressed.size(), compressSeconds, decompressSeconds);
      }
      return 0;
   } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 1;
   }
}
