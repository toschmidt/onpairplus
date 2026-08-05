#include "OnPairPlus.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <span>
#include <string>
#include <vector>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace {
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
void writeFile(const std::string& path, const std::vector<std::byte>& data) {
   std::ofstream file(path, std::ios::binary | std::ios::trunc);
   if (!file) throw std::runtime_error("cannot open output file: " + path);
   file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
   if (!file) throw std::runtime_error("cannot write output file: " + path);
}
//---------------------------------------------------------------------------
}
//---------------------------------------------------------------------------
int main(int argc, char** argv) {
   if (argc != 4 || (std::strcmp(argv[1], "c") != 0 && std::strcmp(argv[1], "d") != 0)) {
      std::cerr << "OnPair+ compressor\n"
                << "Usage: " << argv[0] << " c <input> <output>   compress\n"
                << "       " << argv[0] << " d <input> <output>   decompress\n";
      return 1;
   }

   try {
      const bool compress = std::strcmp(argv[1], "c") == 0;
      const auto input = readFile(argv[2]);
      std::vector<std::byte> output;

      const auto begin = std::chrono::steady_clock::now();
      if (compress)
         opp::encode(input, output);
      else
         opp::decode(input, output);
      const auto end = std::chrono::steady_clock::now();

      writeFile(argv[3], output);

      const double seconds = std::chrono::duration<double>(end - begin).count();
      const double payload = static_cast<double>(compress ? input.size() : output.size());
      const double mibPerSec = seconds > 0.0 ? payload / (1024.0 * 1024.0) / seconds : 0.0;
      if (compress)
         std::fprintf(stderr, "compressed %zu -> %zu bytes (ratio %.3f) in %.3fs (%.1f MiB/s)\n",
                      input.size(), output.size(),
                      output.empty() ? 0.0 : static_cast<double>(input.size()) / static_cast<double>(output.size()),
                      seconds, mibPerSec);
      else
         std::fprintf(stderr, "decompressed %zu -> %zu bytes in %.3fs (%.1f MiB/s)\n",
                      input.size(), output.size(), seconds, mibPerSec);
      return 0;
   } catch (const std::exception& e) {
      std::cerr << "error: " << e.what() << "\n";
      return 1;
   }
}
