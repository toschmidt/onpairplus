#pragma once
//---------------------------------------------------------------------------
#include "util/Hash.hpp"
#include <functional>
#include <memory_resource>
#ifdef OPP_BOOST_UNORDERED
#include <boost/unordered/unordered_flat_map.hpp>
#else
#include "unordered_dense/unordered_dense.h"
#endif
//---------------------------------------------------------------------------
// OnPairPlus - standalone OnPair+ string compressor
// (c) 2026 Tobias Schmidt, Nicolas Schmitt - MIT License
//---------------------------------------------------------------------------
namespace opp {
//---------------------------------------------------------------------------
/// The hash map used by the training path. Training is dominated by hash lookups (the streaming
/// merge pass probes the per-length tables at every input position), so an open-addressing map
/// beats std::unordered_map's node-based layout by a wide margin: on dbtext, encoding runs at
/// 64 MiB/s with std::unordered_map, 79 with unordered_dense and 96 with boost's flat map.
/// Both allocate from a memory resource (std::pmr). The choice never reaches the output - finalize() sorts its entries by (length, id) before
/// building the static tables, so map iteration order cannot influence the compressed stream.
#ifdef OPP_BOOST_UNORDERED
template <class K, class V, class H = Hasher> using TrainMap = boost::unordered_flat_map<K, V, H, std::equal_to<K>, std::pmr::polymorphic_allocator<std::pair<const K, V>>>;
#else
template <class K, class V, class H = Hasher> using TrainMap = ankerl::unordered_dense::pmr::map<K, V, H>;
#endif
//---------------------------------------------------------------------------
}
