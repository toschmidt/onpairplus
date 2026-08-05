#include "compressor/Compressor.hpp"
//---------------------------------------------------------------------------
// Definitions for the Compressor base-class virtuals that live in token-vldb2026's Compressor.cpp,
// which we cannot compile here because it pulls in every compressor of that codebase.
//
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace sgtt::compressor {
//---------------------------------------------------------------------------
std::string Compressor::getFileName() { return getName(); }
std::string Compressor::getInfo() { return "{}"; }
void Compressor::printInfo() {}
//---------------------------------------------------------------------------
}
