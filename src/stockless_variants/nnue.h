// Stockless Variants contributors, 2026. SPDX-License-Identifier: AGPL-3.0-only
// Independent SVNNUE-v1 backend. This is not a Stockfish/FSF network format.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "../types.h"

namespace Stockfish {
class Position;
struct Variant;
}

namespace Stockfish::Stockless::Neural {

constexpr std::uint32_t FormatVersion = 1;
constexpr std::uint32_t FeatureVersion = 1;
constexpr std::uint32_t V2FormatVersion = 2;
constexpr std::uint32_t V2FeatureVersion = 2;
constexpr std::uint32_t Hidden = 64;
constexpr std::uint32_t QA = 256;
constexpr std::uint32_t QB = 64;
constexpr std::uint32_t HeaderBytes = 160;
constexpr std::int32_t MaximumFeatureMass = 512;
constexpr std::int32_t MaximumWhiteCp = 10000;

// Position::set zeroes this POD together with the other position data. A zero
// generation is invalid. Keeping the accumulator on Position avoids writing
// shared pre-root StateInfo objects when different search threads evaluate.
struct Accumulator {
    std::array<std::int32_t, Hidden> values;
    std::uint64_t generation;
    std::int32_t feature_mass;
};

struct Probe {
    bool used = false;
    bool compatible = false;
    bool accumulator_matches = false;
    bool refreshed = false;
    std::string reason;
    std::int64_t numerator = 0;
    std::int32_t white_cp = 0;
    std::int32_t stm_cp = 0;
    Value value = VALUE_ZERO;
    std::uint64_t generation = 0;
    std::array<std::int32_t, Hidden> accumulator{};
    std::array<std::int32_t, Hidden> refreshed_accumulator{};
};

bool supported_variant(const Variant* variant);
std::string variant_name(const Variant* variant);
std::string descriptor_json(const Variant* variant, bool v2 = false);
std::vector<std::pair<std::uint32_t, std::int32_t>> sparse_features(const Position& pos, bool v2 = false);

// Load/select only while the search is stopped. Protocol callbacks enforce
// that boundary. Failed loads preserve the previous model and backend.
bool load_network(const std::string& path, const Variant* variant, std::string& error);
bool select_backend(const std::string& name, const Variant* variant, std::string& error);
void variant_changed(const Variant* variant);
const char* backend_name();
std::string loaded_file();
std::string status_json(const Variant* variant);
std::string evaluation_json(const Position& pos);

bool applicable(const Position& pos);
Value evaluate(const Position& pos);
Probe probe(const Position& pos, bool compare_refresh = true);

// Called by the board/reserve primitives for make AND undo. With Classical
// selected an invalid accumulator makes the inline call sites no-ops.
void board_delta(const Position& pos, Piece pc, Square square, int amount);
void pocket_delta(const Position& pos, Piece pc, int amount);

} // namespace Stockfish::Stockless::Neural
