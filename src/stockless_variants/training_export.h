// Copyright (C) 2026 Stockless Variants contributors
// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <sstream>
#include <string>

#include "../apiutil.h"
#include "../movegen.h"
#include "../position.h"
#include "../uci.h"
#include "nnue.h"

namespace Stockfish::Stockless {

inline std::string training_json_string(const std::string& value) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') {
            result += '\\';
            result += char(c);
        } else if (c < 32) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else result += char(c);
    }
    return result + '"';
}

// Export native Position state. No Python interpretation of FEN, piece letters,
// board stride, Shogi promotion identities, or pocket syntax is involved.
// This diagnostic command is deliberately limited to the standard UCI dialect.
inline std::string training_features_json(const Position& pos, std::uint32_t version = 1) {
    if (CurrentProtocol != UCI_GENERAL)
        return "{\"schema\":1,\"error\":\"nnuefeatures requires the uci protocol\"}";

    const std::string descriptor = Neural::descriptor_json(pos.variant(), version);
    if (!Neural::supported_variant(pos.variant()))
        return "{\"schema\":1,\"descriptor\":" + descriptor
             + ",\"error\":\"unsupported training variant\"}";
    if (pos.is_chess960())
        return "{\"schema\":1,\"descriptor\":" + descriptor
             + ",\"error\":\"Chess960 is not an SVNNUE-v1 training target\"}";

    const auto features = Neural::sparse_features(pos, version);
    if (features.empty())
        return "{\"schema\":1,\"descriptor\":" + descriptor
             + ",\"error\":\"position has no valid SVNNUE features\"}";

    const MoveList<LEGAL> legal(pos);
    Value result = VALUE_ZERO;
    bool ended = pos.is_immediate_game_end(result);
    const char* reason = ended ? "immediate_variant" : nullptr;
    if (!ended && legal.size() == 0) {
        ended = true;
        result = pos.checkers() ? pos.checkmate_value() : pos.stalemate_value();
        reason = pos.checkers() ? "checkmate" : "stalemate";
    }
    if (!ended && pos.is_optional_game_end(result)) {
        ended = true;
        reason = "optional_rule";
    }

    // Optional game ends can be wins or losses: perpetual check/chase and Janggi
    // material scoring must retain the signed native result. Insufficient-mate
    // flags are diagnostics, never an unconditional draw adjudication here.
    std::ostringstream out;
    out << "{\"schema\":1,\"descriptor\":" << descriptor
        << ",\"fen\":" << training_json_string(pos.fen())
        << ",\"side_to_move\":" << int(pos.side_to_move())
        << ",\"features\":[";
    bool first = true;
    for (const auto& feature : features) {
        if (!first) out << ',';
        first = false;
        out << '[' << feature.first << ',' << feature.second << ']';
    }
    out << "],\"legal_moves\":[";
    first = true;
    for (const auto& move : legal) {
        if (!first) out << ',';
        first = false;
        out << training_json_string(UCI::move(pos, move));
    }
    out << "],\"terminal\":{\"ended\":" << (ended ? "true" : "false")
        << ",\"result_stm\":";
    if (ended) out << (result > VALUE_ZERO ? 1 : result < VALUE_ZERO ? -1 : 0);
    else out << "null";
    out << ",\"reason\":" << (reason ? training_json_string(reason) : "null")
        << ",\"claim_policy\":\"claim_all_native_optional\"}"
        << ",\"pieces\":[";
    first = true;
    Bitboard occupied = pos.pieces();
    while (occupied) {
        const Square square = pop_lsb(occupied);
        const Piece piece = pos.piece_on(square);
        if (!first) out << ',';
        first = false;
        out << "{\"file\":" << int(file_of(square))
            << ",\"rank\":" << int(rank_of(square))
            << ",\"color\":" << int(color_of(piece))
            << ",\"type\":" << int(type_of(piece))
            << ",\"unpromoted_type\":" << int(type_of(pos.unpromoted_piece_on(square)))
            << ",\"promoted\":" << (pos.is_promoted(square) ? "true" : "false") << '}';
    }
    out << "],\"pockets\":[";
    first = true;
    for (Color color : {WHITE, BLACK})
        for (PieceSet types = pos.piece_types(); types;) {
            const PieceType type = pop_lsb(types);
            const int count = pos.count_in_hand(color, type);
            if (!count) continue;
            if (!first) out << ',';
            first = false;
            out << "{\"color\":" << int(color) << ",\"type\":" << int(type)
                << ",\"count\":" << count << '}';
        }
    out << "],\"insufficient_material\":{\"white\":"
        << (has_insufficient_material(WHITE, pos) ? "true" : "false")
        << ",\"black\":" << (has_insufficient_material(BLACK, pos) ? "true" : "false")
        << ",\"adjudicated\":false}}";
    return out.str();
}

} // namespace Stockfish::Stockless
