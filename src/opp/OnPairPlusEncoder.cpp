#include "opp/OnPairPlusEncoder.hpp"
#include "util/TrainMap.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
void OnPairPlusEncoder::trainDictionary(const std::span<const std::byte> rawInput)
// Build an OnPair-dictionary on the input, capping the symbol table at 1/16 of the input size,
// seed the leftover token-id space with n-grams of the most frequent characters, and finalize the
// token ids so that neither the tokenizer nor the stored dictionary needs any id mapping later.
{
   input = {reinterpret_cast<const uint8_t*>(rawInput.data()), rawInput.size()};
   const size_t inputSize = input.size();
   const uint8_t* data = input.data();

   // The merge cutoff below stops training once the symbol table would exceed inputSize/16 bytes
   // at 4 bytes per symbol, so an input can never produce more than 256 + inputSize/64 trained
   // symbols. Sizing the training structures by that bound instead of MAX_TOKEN_COUNT removes a
   // fixed ~70us per encode (allocating and clearing 65536-token tables) that small inputs paid in
   // full. The n-gram block is reserved separately below, once its exact size is known.
   const size_t trainedTokenLimit = std::min<size_t>(MAX_TOKEN_COUNT, 256 + inputSize / 64 + 1);
   tokenizer.reserve(trainedTokenLimit);

   // The offsets to the symbols in the dictionary
   Vector<uint32_t> symbolOffsets(memory);
   symbolOffsets.reserve(trainedTokenLimit);
   symbolOffsets.push_back(0);
   // The symbols
   Vector<uint8_t> symbolDictionary(memory);
   symbolDictionary.reserve(trainedTokenLimit * MAX_SYMBOL_LENGTH);
   // A mapping from a token id to its parent token ids, i.e. the tokens that were merged to create it
   TrainMap<uint16_t, std::pair<uint16_t, uint16_t>> childToParents(memory);
   childToParents.reserve(trainedTokenLimit);

   // The training sample: chunkCount^(3/4) chunks, so the sampled fraction shrinks smoothly as
   // the input grows (tiny inputs are used in full), capped at MAX_TRAIN_CHUNKS to bound the
   // training cost on huge inputs
   const size_t chunkCount = (inputSize + TRAIN_CHUNK_SIZE - 1) / TRAIN_CHUNK_SIZE;
   const size_t sampleChunks = std::min({static_cast<size_t>(std::pow(static_cast<double>(chunkCount), 0.75)), MAX_TRAIN_CHUNKS, chunkCount});
   const size_t sampleSize = std::min(inputSize, sampleChunks * TRAIN_CHUNK_SIZE);

   // Tracks the frequency of token pairs
   TrainMap<std::pair<uint16_t, uint16_t>, uint16_t> frequency(memory);
   // Set the threshold for merging tokens based on the size of the training sample
   const double sampleSizeMiB = static_cast<double>(sampleSize) / (1024.0 * 1024.0);
   const auto threshold = static_cast<size_t>(std::fmax(std::log2(sampleSizeMiB), 2.0));
   // Set a limit to the size of the symbol table to prevent it from dominating the compressed size
   const auto maxTableSize = inputSize / 16;

   uint16_t lastUsedToken = 0;
   for (; lastUsedToken <= std::numeric_limits<uint8_t>::max(); lastUsedToken++) {
      uint64_t value = lastUsedToken; // 8 bytes, since the tokenizer reads 8 bytes during insert
      tokenizer.insert(reinterpret_cast<uint8_t*>(&value), 1, lastUsedToken);
      symbolDictionary.push_back(static_cast<uint8_t>(value));
      symbolOffsets.push_back(symbolDictionary.size());
   }
   // Number of base (length-1) symbols (256). Excluded from the merge cutoff so that the built
   // dictionary does not depend on the fixed base-symbol overhead.
   const size_t baseSymbols = symbolOffsets.size() - 1;

   // Train on the sampled chunks, visited in a seeded-shuffled order (the first sampleChunks of
   // the shuffled order are a uniform random sample), so the sample seen before the table cutoff
   // is drawn evenly from the whole input instead of overfitting its beginning. charFreq (for the
   // n-gram seeding below) is computed on the same sample.
   std::array<uint64_t, 256> charFreq{};
   Vector<uint32_t> shuffledChunks(chunkCount, memory);
   std::iota(shuffledChunks.begin(), shuffledChunks.end(), 0);
   std::mt19937 g(42);
   std::shuffle(shuffledChunks.begin(), shuffledChunks.end(), g);
   shuffledChunks.resize(sampleChunks);

   bool dictionaryFull = false;
   for (const auto chunk : shuffledChunks) {
      if (dictionaryFull) break;

      const uint8_t* iter = data + static_cast<size_t>(chunk) * TRAIN_CHUNK_SIZE;
      const uint8_t* end = std::min(iter + TRAIN_CHUNK_SIZE, data + inputSize);

      // Count the character frequencies of the sampled chunk
      for (const uint8_t* p = iter; p < end; ++p) ++charFreq[*p];

      // First match to get everything started
      auto [prevMatchToken, prevMatchLen] = *tokenizer.findLongestMatch(iter, end - iter);
      iter += prevMatchLen;

      while (iter < end) {
         // Find the longest match
         auto match = tokenizer.findLongestMatch(iter, end - iter);
         assert(match);
         const uint16_t matchToken = match->first;
         const size_t matchLen = match->second;

         bool addedToken = false;
         if (prevMatchLen + matchLen <= MAX_SYMBOL_LENGTH) {
            // Update token frequency and possibly merge tokens
            auto tokenPair = std::make_pair(prevMatchToken, matchToken);
            ++frequency[tokenPair];

            if (frequency[tokenPair] >= threshold) {
               const auto* symbolStart = iter - prevMatchLen;
               const auto symbolLen = prevMatchLen + matchLen;
               addedToken = tokenizer.insert(symbolStart, symbolLen, lastUsedToken);
               if (addedToken) {
                  symbolDictionary.insert(symbolDictionary.end(), symbolStart, symbolStart + symbolLen);
                  symbolOffsets.push_back(symbolDictionary.size());

                  // Remember how the token was created
                  childToParents[lastUsedToken] = {prevMatchToken, matchToken};

                  frequency.erase(tokenPair);
                  prevMatchToken = lastUsedToken;
                  prevMatchLen += matchLen;

                  if ((symbolOffsets.size() - 1 - baseSymbols) * 4 > maxTableSize || lastUsedToken++ == std::numeric_limits<uint16_t>::max()) {
                     dictionaryFull = true;
                     break;
                  }
               }
            }
         }
         if (!addedToken) {
            prevMatchToken = matchToken;
            prevMatchLen = matchLen;
         }

         iter += matchLen;
      }
   }

   // The trained dictionary above is exactly the classic onpair table. Now fill the *leftover*
   // token-id space with n-grams of the most frequent characters, for increasing n (digrams, then
   // trigrams, then fourgrams) while space allows. Each n-gram level seeds every K^n combination of
   // its top-K characters (n bytes/token), so any run tokenizes at up to n bytes/token -- the payload
   // never expands, and higher n gives a better worst-case floor. The n-grams are added *after*
   // training, so they never displace a trained symbol; a per-level 256-bit char bitmap rebuilds them.
   const size_t numTrained = (symbolOffsets.size() - 1) - baseSymbols;
   const size_t used = baseSymbols + numTrained; // 256 + trained

   // Characters ranked by descending sample frequency (most-used first)
   std::array<uint8_t, 256> byOrder{};
   size_t distinct = 0;
   for (unsigned b = 0; b < 256; ++b)
      if (charFreq[b]) byOrder[distinct++] = static_cast<uint8_t>(b);
   // Ties keep the ascending character order (std::sort with the character as tie-break; std::stable_sort
   // would take its buffer from the global heap)
   std::sort(byOrder.begin(), byOrder.begin() + distinct, [&](uint8_t a, uint8_t b) { return charFreq[a] > charFreq[b] || (charFreq[a] == charFreq[b] && a < b); });
   double sampleBytes = 0;
   for (unsigned b = 0; b < 256; ++b) sampleBytes += static_cast<double>(charFreq[b]);

   auto lengthOf = [&](size_t id) { return symbolOffsets[id + 1] - symbolOffsets[id]; };

   // Build the seeded, finalized dictionary for a token-id budget of `cap`. The merge sequence is
   // shared: a smaller budget simply keeps its prefix. When the budget admits every trained symbol
   // the tokenizer built during training is moved in, otherwise the symbols that fit are
   // re-inserted into a fresh table.
   auto buildCandidate = [&](const size_t cap, Candidate& candidate, const bool reuseTrainingTables) {
      const size_t usedC = std::min(used, cap);

      // Greedily allocate the leftover token space to increasing n. For each level, K is the largest
      // char count (<= distinct) with K^n fitting the remaining budget; insert all K^n n-grams with
      // contiguous ids just past the previous level (collisions with a trained symbol just fail).
      // A level additionally only admits characters whose rarest n-gram (the K-th most frequent char
      // in every position, under an independence estimate) is expected at least threshold^2 times in
      // the sample -- pruning never-occurring combinations of rare characters frees budget for larger
      // higher-order alphabets and keeps the decoder's table small.
      candidate.levels.clear();
      size_t base = baseSymbols;
      size_t budget = usedC < cap ? cap - usedC : 0;
      for (unsigned n = 2; n <= MAX_NGRAM_ORDER; ++n) {
         size_t K = std::min<size_t>(distinct, static_cast<size_t>(std::pow(static_cast<double>(budget), 1.0 / n)));
         auto fits = [&](size_t k) { size_t c = 1; for (unsigned t = 0; t < n; ++t) { c *= k; if (c > budget) return false; } return true; };
         while (K > 0 && !fits(K)) --K;
         while (K > 0 && sampleBytes * std::pow(static_cast<double>(charFreq[byOrder[K - 1]]) / sampleBytes, n) < static_cast<double>(threshold * threshold)) --K;
         if (K == 0) break; // no room; higher n needs even more

         NgramLevel L;
         L.n = n;
         L.K = K;
         L.base = base;
         L.idx.fill(-1);
         for (size_t i = 0; i < K; ++i) {
            const uint8_t c = byOrder[i];
            L.bitmap[c >> 3] |= static_cast<uint8_t>(1u << (c & 7));
         }
         size_t cc = 0;
         for (unsigned b = 0; b < 256; ++b)
            if (L.bitmap[b >> 3] & (1u << (b & 7))) {
               L.chars[cc] = static_cast<uint8_t>(b);
               L.idx[b] = static_cast<int16_t>(cc);
               ++cc;
            }
         const size_t count = L.count();
         base += count;
         budget -= count;
         candidate.levels.push_back(L);
      }
      const size_t totalNgrams = base - baseSymbols;

      // The levels are laid out; only now is the n-gram block's size known, so grow the tokenizer's
      // tables once to their final size and seed the combinations with contiguous ids in level order
      if (reuseTrainingTables && usedC == used) {
         candidate.tokenizer = std::move(tokenizer);
      } else {
         candidate.tokenizer.reserve(usedC + totalNgrams);
         for (size_t id = baseSymbols; id < usedC; ++id)
            candidate.tokenizer.insert(symbolDictionary.data() + symbolOffsets[id], lengthOf(id), static_cast<uint16_t>(id));
      }
      candidate.tokenizer.reserve(usedC + totalNgrams);
      size_t tempId = usedC;
      for (const auto& L : candidate.levels)
         forEachNgram(L, [&](uint64_t value) {
            candidate.tokenizer.insert(reinterpret_cast<uint8_t*>(&value), L.n, static_cast<uint16_t>(tempId++));
         });

      // Finalize the token ids. The final id order is [256 singles][all n-grams, low n first][kept
      // trained, length-sorted]. Every trained symbol of length 2..MAX_NGRAM_ORDER that is *also* a
      // seeded n-gram is dropped -- it is reconstructable from the bitmaps -- and any reference to it
      // is remapped to that n-gram's id. The remap is applied to the tokenizer and to the recipe
      // parent ids right here and then discarded, so no id mapping survives training.
      auto asNgram = [&](size_t id) -> int64_t {
         const size_t l = lengthOf(id);
         if (l < 2 || l > MAX_NGRAM_ORDER) return -1;
         for (const auto& L : candidate.levels)
            if (L.n == l) return ngramIdOf(L, symbolDictionary.data() + symbolOffsets[id], l);
         return -1;
      };

      // Kept trained ids (all trained except the n-gram duplicates), length-sorted so parents precede
      Vector<uint16_t> kept(memory);
      kept.reserve(usedC - baseSymbols);
      for (size_t id = baseSymbols; id < usedC; ++id)
         if (asNgram(id) < 0) kept.push_back(static_cast<uint16_t>(id));
      // Ties keep the ascending id order kept was filled in (std::sort with the id as tie-break)
      std::sort(kept.begin(), kept.end(), [&](uint16_t a, uint16_t b) { return lengthOf(a) < lengthOf(b) || (lengthOf(a) == lengthOf(b) && a < b); });

      // Final id of every training id in [0, usedC + totalNgrams): singles keep their id, a dropped
      // n-gram duplicate maps to its n-gram id, each kept trained symbol lands just past the whole
      // n-gram block, and the contiguous n-gram temp ids (usedC + m) map straight to their final ids
      // (256 + m), since both were laid out in the same order.
      Vector<uint16_t> remap(usedC + totalNgrams, memory);
      for (size_t k = 0; k < baseSymbols; ++k) remap[k] = static_cast<uint16_t>(k);
      for (size_t p = 0; p < kept.size(); ++p) remap[kept[p]] = static_cast<uint16_t>(baseSymbols + totalNgrams + p);
      for (size_t id = baseSymbols; id < usedC; ++id) {
         const int64_t ng = asNgram(id);
         if (ng >= 0) remap[id] = static_cast<uint16_t>(ng);
      }
      for (size_t m = 0; m < totalNgrams; ++m) remap[usedC + m] = static_cast<uint16_t>(baseSymbols + m);

      // Build the finalized dictionary blocks in final id order: the length histogram (256 implicit
      // singles plus the kept trained symbols), the raw bytes of the kept symbols of length
      // 2..HYBRID_SHORT_MAX, and the merge recipe (two parent final-ids) of every longer kept symbol
      candidate.lengthHistogram.fill(0);
      candidate.lengthHistogram[0] = static_cast<uint16_t>(baseSymbols);
      for (const uint16_t id : kept) ++candidate.lengthHistogram[lengthOf(id) - 1];

      candidate.shortSymbols.clear();
      candidate.recipes.clear();
      for (const uint16_t id : kept) {
         const uint32_t len = lengthOf(id);
         if (len > HYBRID_SHORT_MAX) break;
         candidate.shortSymbols.insert(candidate.shortSymbols.end(), symbolDictionary.begin() + symbolOffsets[id], symbolDictionary.begin() + symbolOffsets[id + 1]);
      }
      for (const uint16_t id : kept) {
         if (lengthOf(id) <= HYBRID_SHORT_MAX) continue;
         const auto it = childToParents.find(id);
         assert(it != childToParents.end());
         candidate.recipes.push_back(remap[it->second.first]);
         candidate.recipes.push_back(remap[it->second.second]);
      }

      // Rewrite the tokenizer's stored ids to final ids, so compress() emits final ids directly
      candidate.tokenizer.remapIds(remap);
      // Build the static compression tables used by compress(); trained symbols start right past
      // the n-gram block and take priority in the lossy 3-byte map
      candidate.tokenizer.finalize(static_cast<uint16_t>(baseSymbols + totalNgrams));
      candidate.dictionaryTokens = baseSymbols + totalNgrams + kept.size();
   };

   // Tokenize the head of the input with a candidate's tables and count the ids it produces
   auto countTokens = [&](const Tokenizer& candidate) {
      const size_t evaluationBytes = std::min({inputSize, sampleSize, WIDTH_EVAL_SIZE});
      const uint8_t* iter = data;
      const uint8_t* const end = data + evaluationBytes;
      const uint8_t* const denseEnd = end - std::min<size_t>(evaluationBytes, MAX_SYMBOL_LENGTH - 1);
      uint64_t count = 0;
      while (iter < denseEnd) {
         iter += candidate.match(iter).second;
         ++count;
      }
      while (iter < end) {
         iter += candidate.match(iter, end - iter).second;
         ++count;
      }
      return count;
   };

   // A dictionary of at most MAX_TOKEN_COUNT_12 tokens is emitted with 12-bit ids, which costs a
   // quarter less per id but yields more ids because the table is smaller. Build both candidates
   // from the same merge sequence, tokenize the evaluation prefix with each, and keep whichever
   // spends fewer bits. Below the threshold the full table cannot be reached anyway, so only the
   // small candidate is built.
   Candidate candidate12(memory);
   Candidate candidateFull(memory);
   tokenCount12Value = 0;
   tokenCount16Value = 0;
   bool use12 = true;
   // Below the threshold the merge cutoff cannot produce more than MAX_TOKEN_COUNT_12 trained
   // symbols, so the full table is unreachable and only the small candidate is worth building.
   comparedFullTableValue = trainedTokenLimit > MAX_TOKEN_COUNT_12;
   if (comparedFullTableValue) {
      // The full candidate keeps the tables training already filled; the small one re-inserts the
      // prefix of the merge sequence that fits into its own table
      buildCandidate(MAX_TOKEN_COUNT, candidateFull, true);
      buildCandidate(MAX_TOKEN_COUNT_12, candidate12, false);
      tokenCount16Value = countTokens(candidateFull.tokenizer);
      tokenCount12Value = countTokens(candidate12.tokenizer);
      use12 = tokenCount12Value * 12 <= tokenCount16Value * 16;
   } else {
      buildCandidate(MAX_TOKEN_COUNT_12, candidate12, true);
   }

   Candidate& chosen = use12 ? candidate12 : candidateFull;
   tokenizer = std::move(chosen.tokenizer);
   levels = std::move(chosen.levels);
   lengthHistogram = chosen.lengthHistogram;
   shortSymbols = std::move(chosen.shortSymbols);
   recipes = std::move(chosen.recipes);
   // The decoder derives the width from the dictionary size, so the encoder must follow the same
   // rule rather than the comparison above
   tokenBitsValue = chosen.dictionaryTokens <= MAX_TOKEN_COUNT_12 ? 12 : 16;
}
//---------------------------------------------------------------------------
size_t OnPairPlusEncoder::serializeDictionary(const std::span<std::byte> out) const
// Serialize the finalized dictionary: [MAX_NGRAM_ORDER-1 char bitmaps][16-entry length histogram]
// [short raw bytes][long recipes]. Each level writes its 32-byte bitmap (missing levels stay
// zero); decode derives each level's K and base from the bitmaps. Returns the serialized size.
{
   assert(out.size() >= MAX_DICTIONARY_SIZE);

   std::memset(out.data(), 0, HEADER_BYTES);
   for (const auto& L : levels)
      std::memcpy(out.data() + (L.n - 2) * BITMAP_BYTES, L.bitmap.data(), BITMAP_BYTES);

   auto* writer = out.data() + HEADER_BYTES;
   constexpr size_t histSize = MAX_LENGTHS * sizeof(uint16_t);
   std::memcpy(writer, lengthHistogram.data(), histSize);
   writer += histSize;
   std::memcpy(writer, shortSymbols.data(), shortSymbols.size());
   writer += shortSymbols.size();
   std::memcpy(writer, recipes.data(), recipes.size() * sizeof(uint16_t));
   writer += recipes.size() * sizeof(uint16_t);

   assert(static_cast<size_t>(writer - out.data()) <= out.size());
   return writer - out.data();
}
//---------------------------------------------------------------------------
namespace {
//---------------------------------------------------------------------------
/// Sinks that append token ids at a fixed physical width. Packing is fused into the match loop
/// rather than run as a second pass over the finished payload: the loop is a serial dependency
/// chain waiting on the tokenizer's hash probes, so the shift and the (well-predicted) flush
/// branch hide in that latency, while a separate pass would re-read and rewrite every byte.
template <unsigned Bits>
struct TokenWriter;
//---------------------------------------------------------------------------
template <>
struct TokenWriter<16> {
   uint16_t* writer;

   explicit TokenWriter(std::byte* out) : writer(reinterpret_cast<uint16_t*>(out)) {}
   void emit(const uint16_t id) { *writer++ = id; }
   size_t finish(const std::byte* out) { return reinterpret_cast<std::byte*>(writer) - out; }
};
//---------------------------------------------------------------------------
template <>
struct TokenWriter<12> {
   uint8_t* writer;
   /// The pending low bits, least-significant id first
   uint64_t buffer = 0;
   /// The number of valid bits in the buffer (always < 32 between ids)
   unsigned bits = 0;

   explicit TokenWriter(std::byte* out) : writer(reinterpret_cast<uint8_t*>(out)) {}
   void emit(const uint16_t id) {
      assert(id < MAX_TOKEN_COUNT_12);
      buffer |= static_cast<uint64_t>(id) << bits;
      bits += 12;
      if (bits >= 32) {
         unalignedStore(writer, static_cast<uint32_t>(buffer));
         writer += 4;
         buffer >>= 32;
         bits -= 32;
      }
   }
   size_t finish(const std::byte* out) {
      // Flush byte-exactly: n ids occupy ceil(12n/8) bytes, which is what the decoder inverts to
      // recover the id count, so no padding may be written here
      while (bits) {
         *writer++ = static_cast<uint8_t>(buffer);
         buffer >>= 8;
         bits -= std::min(bits, 8u);
      }
      return reinterpret_cast<std::byte*>(writer) - out;
   }
};
//---------------------------------------------------------------------------
template <unsigned Bits>
size_t tokenize(const Tokenizer& tokenizer, const std::span<const uint8_t> input, const std::span<std::byte> out) {
   TokenWriter<Bits> writer(out.data());
   const uint8_t* iter = input.data();
   const uint8_t* end = iter + input.size();

   // Dense loop: every position has at least 16 readable bytes, so match() runs on plain loads
   // with no length checks
   const uint8_t* denseEnd = end - std::min<size_t>(input.size(), MAX_SYMBOL_LENGTH - 1);
   while (iter < denseEnd) {
      const auto [token, symbolLen] = tokenizer.match(iter);
      writer.emit(token);
      iter += symbolLen;
   }

   // Tail loop: the last few bytes go through the length-aware match(), which zero-extends them
   // into a local 16-byte buffer
   while (iter < end) {
      const auto [token, symbolLen] = tokenizer.match(iter, end - iter);
      writer.emit(token);
      iter += symbolLen;
   }

   return writer.finish(out.data());
}
//---------------------------------------------------------------------------
}
//---------------------------------------------------------------------------
size_t OnPairPlusEncoder::compress(const std::span<std::byte> out) const
// Tokenize the input into a stream of final token ids using the static compression tables,
// emitting them at the width chosen by trainDictionary(). Returns the number of bytes written.
{
   return tokenBitsValue == 12 ? tokenize<12>(tokenizer, input, out) : tokenize<16>(tokenizer, input, out);
}
//---------------------------------------------------------------------------
}
