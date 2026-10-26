// Stockless Variants contributors, 2026. SPDX-License-Identifier: AGPL-3.0-only
// Adapted from Stockless v0.9.0 threat_signal.h at the revision in upstream.lock.json.
// Geometry and attack semantics are supplied by the GPL-licensed FSF core.
#pragma once

#include <algorithm>
#include "../bitboard.h"
#include "../position.h"
#include "../types.h"
#include "config.h"

namespace Stockfish::Stockless {

struct ThreatSignal {
    int attackPressure = 0;
    int kingDanger = 0;
    int tacticalVolatility = 0;
};

inline Bitboard attacks_for(const Position& pos, Color c) noexcept {
    Bitboard result = 0;
    Bitboard occupied = pos.pieces(c) & ~pos.pieces(c, KING);
    while (occupied)
    {
        const Square s = pop_lsb(occupied);
        // This API includes palace diagonals, cannon screens, horse/elephant
        // blockers, pawn direction, river restrictions and custom Betza pieces.
        result |= pos.attacks_from(c, type_of(pos.piece_on(s)), s);
    }
    return result & pos.board_bb();
}

inline Bitboard royal_zone(const Position& pos, Color c) noexcept {
    const PieceType royal = pos.blast_on_capture() && pos.extinction_pseudo_royal()
                          && pos.extinction_piece_types() == piece_set(COMMONER)
                          ? COMMONER : KING;
    if (pos.count(c, royal) != 1)
        return 0;
    const Square k = pos.square(c, royal);
    // A geometric contact zone, not a legal king-move generator.
    return (attacks_bb<KING>(k) | k) & pos.board_bb();
}

inline ThreatSignal compute_threat_signal(const Position& pos, int depth,
                                          bool importantNode,
                                          const SearchSettings& settings) noexcept {
    ThreatSignal out{};
    if (!settings.hybrid || !settings.threats
        || (depth < (settings.mobile ? 7 : 5) && !importantNode))
        return out;

    const Color us = pos.side_to_move(), them = ~us;
    const Bitboard ourAttacks = attacks_for(pos, us);
    const Bitboard theirAttacks = attacks_for(pos, them);
    const Bitboard ourZone = royal_zone(pos, us);
    const Bitboard theirZone = royal_zone(pos, them);
    const int contactsFor = popcount(ourAttacks & theirZone);
    const int contactsAgainst = popcount(theirAttacks & ourZone);
    const int attackedEnemy = popcount(ourAttacks & pos.pieces(them));
    const int attackedUs = popcount(theirAttacks & pos.pieces(us));
    const int pinsFor = popcount(pos.pinners(us));
    const int pinsAgainst = popcount(pos.pinners(them));

    out.attackPressure = 36 * contactsFor + 12 * attackedEnemy + 24 * pinsFor;
    out.kingDanger = 42 * contactsAgainst + 12 * attackedUs + 28 * pinsAgainst;
    out.tacticalVolatility = 14 * (attackedEnemy + attackedUs)
                          + 20 * (pinsFor + pinsAgainst)
                          + 12 * (contactsFor + contactsAgainst);

    if (pos.blast_on_capture())
    {
        // Adjacent capturable pieces can transmit a blast to a king. This is
        // a bounded search hint; legality and explosions remain FSF's rules.
        out.attackPressure += 32 * popcount(ourAttacks & pos.pieces(them) & theirZone);
        out.kingDanger += 40 * popcount(theirAttacks & pos.pieces(us) & ourZone);
    }

    if (pos.piece_drops() && pos.drop_checks())
    {
        int checkingDrops = 0;
        for (PieceSet remaining = pos.piece_types(); remaining;)
        {
            const PieceType pt = pop_lsb(remaining);
            if (pos.count_in_hand(us, pt) > 0)
                checkingDrops += popcount(pos.check_squares(pt) & ~pos.pieces()
                                         & pos.drop_region(us, pt));
        }
        // Candidate checks may include illegal drops (nifu/uchifuzume).
        // They influence only a capped ordering/depth hint, never move legality.
        out.attackPressure += 6 * std::min(checkingDrops, 16);
        out.tacticalVolatility += 8 * std::min(16, pos.count_in_hand(ALL_PIECES));
    }

    out.attackPressure = std::clamp(out.attackPressure, 0, 256);
    out.kingDanger = std::clamp(out.kingDanger, 0, 256);
    out.tacticalVolatility = std::clamp(out.tacticalVolatility, 0, 256);
    return out;
}

} // namespace Stockfish::Stockless
