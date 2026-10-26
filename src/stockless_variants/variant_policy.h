// Stockless Variants contributors, 2026. SPDX-License-Identifier: AGPL-3.0-only
// Variant safety and fixed-point adaptation for Stockless v0.9.0's policy.
#pragma once

#include <algorithm>
#include <cstdint>
#include "../position.h"
#include "config.h"

namespace Stockfish::Stockless {

struct VariantSearchPolicy {
    const char* family;
    bool hybrid;
    bool overdrive;
    bool nullMove;
};

inline VariantSearchPolicy variant_search_policy(const Position& pos) noexcept {
    const Variant& v = *pos.variant();
    // These rules need objective/turn-aware signals, not a royal attack score.
    // They retain FSF's native search, with conservative null-move gating below.
    if (v.twoBoards)
        return {"partner-board", false, false, false};
    if (v.wallingRule != NO_WALLING)
        return {"compound-turn", false, false, false};
    if (v.mustCapture || v.extinctionValue == VALUE_MATE)
        return {"compulsory-capture", false, false, false};
    // FSF represents orthodox Atomic kings as pseudo-royal COMMONER pieces.
    if (v.blastOnCapture && v.extinctionPseudoRoyal
        && v.extinctionPieceTypes == piece_set(COMMONER)
        && pos.count<COMMONER>(WHITE) == 1 && pos.count<COMMONER>(BLACK) == 1)
        return {"atomic", true, false, false};
    if (!v.checking || v.connectN || v.connectNxN || v.collinearN
        || v.flipEnclosedPieces || v.flagRegion[WHITE] || v.flagRegion[BLACK]
        || pos.count<KING>(WHITE) != 1 || pos.count<KING>(BLACK) != 1)
        return {"alternative-objective", false, false, false};
    if (v.blastOnCapture)
        return {"atomic", true, false, false};
    if (v.diagonalLines && (v.pieceTypes & JANGGI_CANNON))
        return {"janggi", true, false, false};
    if (v.flyingGeneral)
        return {"xiangqi", true, false, true};
    if (v.pieceDrops || v.capturesToHand)
        return {v.shogiStylePromotions ? "shogi-drops" : "pocket-drops", true, false, false};
    if (v.shogiStylePromotions)
        return {"shogi", true, false, true};
    if (v.checkCounting)
        return {"check-counting", true, false, false};
    const bool orthodox = v.pieceTypes == CHESS_PIECES && v.maxFile == FILE_H
                       && v.maxRank == RANK_8 && !v.gating
                       && !v.pass[WHITE] && !v.pass[BLACK];
    return {orthodox ? "chess" : "regional-fairy", true, orthodox, true};
}

inline SearchSettings variant_search_settings(const Position& pos,
                                              SearchSettings settings) noexcept {
    if (!settings.hybrid)
        return settings;
    const auto policy = variant_search_policy(pos);
    settings.hybrid = settings.hybrid && policy.hybrid;
    settings.threats = settings.threats && settings.hybrid;
    settings.overdrive = settings.overdrive && policy.overdrive;
    return settings;
}

// Stockless uses 1/1024 ply; this FSF revision uses integral Depth.
// Quantize the bounded sub-ply buyback using a stable position/move selector.
// Plain truncation erases all signals; nearest rounding erases the entire
// mobile profile (cap=320). Stateless threshold quantization retains the
// fractional magnitude across positions, including on Android, without a
// shared residual, RNG state or schedule-dependent result. It is an explicit
// adaptation, not bit-identical reproduction of Stockfish 19's search.
inline int adapted_reduction(int nativeReduction, int adjustment,
                             std::uint64_t discriminator) noexcept {
    const int buyback = -std::clamp(adjustment, -832, 0);
    if (!buyback)
        return nativeReduction;
    // SplitMix64 finalizer: disperse Zobrist/move bits before the 10-bit choice.
    discriminator ^= discriminator >> 30;
    discriminator *= UINT64_C(0xbf58476d1ce4e5b9);
    discriminator ^= discriminator >> 27;
    discriminator *= UINT64_C(0x94d049bb133111eb);
    discriminator ^= discriminator >> 31;
    return nativeReduction - int((discriminator & 1023) < std::uint64_t(buyback));
}

} // namespace Stockfish::Stockless
