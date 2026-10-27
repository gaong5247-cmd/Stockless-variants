// Stockless Variants 2026. SPDX-License-Identifier: AGPL-3.0-only
// SVNNUE-v3 Lite: separate experimental Half(K)A-style bucketed evaluator.
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

namespace Stockfish::Stockless::Neural::V3 {

constexpr unsigned Hidden = 128;
constexpr unsigned L1 = 32;
constexpr unsigned L2 = 16;
constexpr unsigned KingBuckets = 16;
constexpr unsigned QA = 256;
constexpr unsigned QB = 128;
constexpr unsigned HeaderSize = 160;

// Position::set zeroes this trivially-copyable state. Independent on each
// Position, so no shared pre-root StateInfo or cross-thread mutation.
struct Accumulator {
    std::array<std::array<std::int32_t, Hidden>, 2> values;
    std::uint64_t generation;
    std::array<std::int32_t, 2> mass;
    std::array<std::uint32_t, 2> bucket;
};

bool supported(const Variant* variant);
bool load(const std::string& path, const Variant* variant, std::string& error);
bool compatible(const Variant* variant);
bool available();
void activate(bool enabled);
void invalidate();
std::string file();
std::string descriptor_json(const Variant* variant);
std::vector<std::pair<std::uint32_t, std::int32_t>>
features(const Position& pos, Color perspective);
bool applicable(const Position& pos);
Value evaluate(const Position& pos);
std::string status_json(const Variant* variant);
std::string evaluation_json(const Position& pos);
std::string native_features_json(const Position& pos);
void board_delta(const Position& pos, Piece piece, Square square, int amount);
void pocket_delta(const Position& pos, Piece piece, int amount);

}  // namespace Stockfish::Stockless::Neural::V3
