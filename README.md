# OnPairPlus

A standalone, dependency-free implementation of **OnPair+**, a dictionary-based string compressor
optimized for very fast decompression at compression ratios close to zstd.

## The algorithm

**OnPair** trains a dictionary of up to 65,536 symbols (each at most 16 bytes) in a *single
streaming pass*: the input is scanned with greedy longest-prefix matching against the current
dictionary, adjacent match pairs are counted, and once a pair reaches a frequency threshold it is
merged into a new symbol that is immediately available for further matching (an online variant of
byte-pair encoding). The compressed stream is simply a sequence of 16-bit token ids; decompression
is a table lookup plus a 16-byte store per token, which makes it extremely fast.

**OnPair+** adds two ideas on top:

- **N-gram seeding**: after training, the leftover token-id space is filled with *all* K^n
  combinations of the K most frequent characters, for n = 2, 3, 4. Any run of those characters
  therefore tokenizes at n bytes per 2-byte token, giving a worst-case expansion floor even on
  data the trained dictionary does not cover. Each level is serialized as just a 32-byte character
  bitmap; the decoder re-enumerates the combinations.
- **Hybrid dictionary serialization**: trained symbols of 2..4 bytes are stored as raw bytes,
  longer ones as a 4-byte *merge recipe* (the two parent token ids); the decoder reconstructs the
  symbol table by concatenating parents, which are guaranteed to precede their children.

## Our changes

This repository ports OnPair+ out of its research codebase and evolves it:

- **Plain blob API** (`opp::encode` / `opp::decode` over byte spans) and a minimal, self-delimiting
  stream format: `[uncompressedSize u64][dictionary][u16 token stream]` — 8 bytes of metadata total.
- **Final ids during training**: the length-sort, n-gram dedup, and id assignment are folded into
  the training phase (the tokenizer's tables are rewritten once), so no remap of the token stream
  and no stored id mapping exist.
- **Bounded, evenly drawn training sample**: the input is split into 64 KiB chunks visited in
  seeded-shuffled order; training sees chunkCount^(3/4) chunks (capped at 128 MiB), so dictionary
  cost is bounded and does not overfit the input's beginning. The merge threshold derives from the
  sample size.
- **Frequency-aware n-gram admission**: an n-gram level only admits characters whose rarest
  combination is expected at least threshold² times, pruning never-occurring combinations in favor
  of larger higher-order alphabets and a smaller decode table.
- **SGTT-style compression tables** (adopted from the token-vldb2026 tokenizer): a 65,536-entry
  direct-lookup array resolving the 2-byte/1-byte tail without hashing, a lossy two-probe map for
  3-byte symbols (with priority insertion so collisions drop the least valuable symbol), and
  static multi hash maps — keyed by the first 4 bytes for 4..8-byte symbols and by the first
  8 bytes for 9..16-byte symbols — with bucket entries sorted longest-first. This took encoding
  from ~35 MiB/s to ~150-350 MiB/s.
- **Zero-copy, overread-free encoding**: a dense match loop with plain 8-byte loads runs up to the
  last 15 bytes; a length-aware tail match zero-extends the rest. No padded input copy, no reads
  past the input span. Symbols may over-match their own zero bytes past the input's end; the
  decoder's final truncation removes them again.
- Exact-sized decoder tables, nested-loop n-gram enumeration, and an uninitialized-by-design
  symbol table keep the per-stream decode setup small.
- **Overlapped table probes**: the match loop is a serial dependency chain (the next position is
  only known once the current match resolves), so both hash directories are looked up before
  either bucket is walked and their cache misses overlap.
- **Input-sized training structures**: the merge cutoff bounds an input to 256 + inputSize/64
  trained symbols, so the training tables are reserved for that instead of the full 65,536, and
  the n-gram block is reserved once its exact size is known. This cut the fixed per-encode cost
  from ~86 us to ~26 us, which dominated small inputs.
- **Open-addressing training maps**: training is hash-lookup bound, so it uses
  `boost::unordered_flat_map` when available and the vendored `ankerl::unordered_dense` otherwise
  (dbtext encoding: 96 / 79 / 64 MiB/s for boost / unordered_dense / `std::unordered_map`). The
  choice cannot reach the output, since `finalize()` sorts its entries before building the static
  tables.

## Build, test, CLI

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build            # Catch2 test suite
./build/opp c <input> <output>    # compress
./build/opp d <input> <output>    # decompress
```

The library itself needs no installed dependencies: `thirdparty/` vendors the single header it
falls back to. If boost's headers are present they are used for the training map (faster, same
output); `-DOPP_USE_BOOST=OFF` forces the vendored one.

```cpp
#include "OnPairPlus.hpp"
std::vector<std::byte> compressed, restored;
opp::encode(input, compressed);
opp::decode(compressed, restored);
```

## Benchmark

`benchmark/` is a self-contained comparison harness (own CMake project; lz4, zstd, brotli via
vcpkg, upstream FSST built from source; all in-memory library calls, best-of-N, roundtrip
verified). The rest of the repository has no dependencies.

Two of the competitors are the direct ancestors of this implementation, built from the sources
vendored in a sibling `token-vldb2026` checkout (point `-DTOKEN_VLDB_DIR=` elsewhere if needed):

- **onpair / onpair16** &mdash; the reference OnPair implementation (Gargiulo and Venturini,
  arXiv:2508.02280), unbounded and 16-byte symbol lengths. `onpair16` is the like-for-like
  baseline for opp, so the gap between them is what the changes listed above buy.
- **tokenizer** &mdash; the SGTT tokenizer, a fixed OpenAI-100k token table with no training at
  all, which bounds how fast a 16-bit token decoder can go.

Both compress the whole input as a single string, like every other compressor here, and both
count their dictionary against the compressed size, exactly as opp's stream format does.

```sh
./benchmark.py dbtext        # or: bitext, wiki9; downloads datasets into ./data
```

## Results

Apple M-class laptop, single-threaded, in-memory, best-of-3. Levels: lz4 default, zstd -3,
brotli -q 5.

**enwik9** (1 GB English Wikipedia):

| compressor | ratio | enc MiB/s | dec MiB/s |
|------------|------:|----------:|----------:|
| lz4        | 1.96  |     691   |    4743   |
| fsst       | 1.65  |     418   |    4307   |
| zstd       | 3.21  |     322   |    1295   |
| brotli     | 3.60  |      79   |     576   |
| tokenizer  | 1.63  |     282   |    8176   |
| onpair     | 2.83  |     103   |    4229   |
| onpair16   | 2.75  |     114   |    5015   |
| **opp**    | **2.82** | **225** | **9256** |
| opp+lz4    | 3.22  |     205   |    5278   |
| opp+zstd   | 3.50  |     162   |    1952   |

opp decompresses fastest of the whole field at a ratio between lz4 and zstd, while the two
static-table baselines (fsst's 255 symbols, the tokenizer's fixed OpenAI-100k table) stay below
ratio 1.7 on this corpus.

Against its own ancestors the picture depends on the input size. On a corpus this large the
reference OnPair reaches the same ratio (2.83 / 2.75 versus opp's 2.82) — a full dictionary is
amortized over a gigabyte, so the bounded table and hybrid serialization buy little. What remains
is speed: opp encodes 2.0x faster than onpair16 and decodes 1.8x faster. The ratio gain shows up
on smaller inputs, where the dictionary is a real share of the output (see dbtext below).

The chained variants run a second general-purpose compressor over opp's token stream: opp+lz4
beats zstd's ratio at 4.1x its decompression speed, and opp+zstd reaches 3.50 — within 3% of
brotli — while decompressing 1.5x faster than plain zstd.

**dbtext** (23 database string columns, 38 MiB) and **bitext** (106 machine-translation corpora,
12.7 GiB), per-file distributions (boxes 25/50/75, whiskers 5/95 percentiles):

![dbtext ratios](plots/dbtext_ratio.png)

![bitext ratios](plots/bitext_ratio.png)

![dbtext encode speed](plots/dbtext_enc.png)

![bitext encode speed](plots/bitext_enc.png)

![dbtext decode speed](plots/dbtext_dec.png)

![bitext decode speed](plots/bitext_dec.png)

Aggregate ratios: dbtext — opp 2.51, opp+zstd 2.77 (zstd 2.77, onpair16 1.91);
bitext — opp 2.80, opp+zstd 3.29 (zstd 2.98, onpair16 2.74). opp decodes at 7-9 GiB/s throughout
and encodes at 89 MiB/s (dbtext) to 135 MiB/s (bitext); the small-file end is dominated by
dictionary training, not by matching.
Per-file CSVs live in `results/`, regenerate the plots with `tools/plot_results.py`.

## License

MIT, (c) 2026 Tobias Schmidt, Nicolas Schmitt. OnPair/OnPair+ originate from the token-vldb2026
and string-compression research codebases.
