// SPDX-License-Identifier: AGPL-3.0-or-later
// Stockless Variants rule regression harness.
// Selected expected positions/results derive from the pinned Fairy-Stockfish
// test.py and tests/perft.sh (GPL-3.0-or-later); see NOTICE.md.
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bitboard.h"
#include "endgame.h"
#include "evaluate.h"
#include "movegen.h"
#include "piece.h"
#include "position.h"
#include "psqt.h"
#include "search.h"
#include "thread.h"
#include "tt.h"
#include "uci.h"
#include "variant.h"

using namespace Stockfish;

namespace {
std::uint64_t checks = 0;
std::uint64_t transitions = 0;
unsigned cases = 0;
unsigned failures = 0;

void require(bool ok, const std::string& explanation) {
    ++checks;
    if (!ok)
        throw std::runtime_error(explanation);
}

void run(const std::string& name, const std::function<void()>& test) {
    ++cases;
    try {
        test();
        std::cout << "PASS " << name << '\n';
    } catch (const std::exception& e) {
        ++failures;
        std::cout << "FAIL " << name << ": " << e.what() << '\n';
    }
}

struct Fixture {
    Position pos;
    std::deque<StateInfo> states;
    const Variant* variant;

    explicit Fixture(const std::string& name, const std::string& fen = "", bool frc = false)
        : states(1), variant(nullptr) {
        auto found = variants.find(name);
        require(found != variants.end(), "unregistered variant: " + name);
        variant = found->second;
        UCI::init_variant(variant);
        PSQT::init(variant);
        pos.set(variant, fen.empty() ? variant->startFen : fen,
                frc || variant->chess960, &states.back(), Threads.main());
    }

    std::set<std::string> moves() const {
        std::set<std::string> result;
        for (Move m : MoveList<LEGAL>(pos))
            result.insert(UCI::move(pos, m));
        return result;
    }

    Move move(std::string notation) {
        Move result = UCI::to_move(pos, notation);
        require(result != MOVE_NONE, "illegal fixture move: " + notation + " in " + pos.fen());
        return result;
    }

    void play(const std::string& notation) {
        Move m = move(notation);
        states.emplace_back();
        pos.do_move(m, states.back());
    }

    void play_sequence(const std::string& sequence) {
        std::istringstream in(sequence);
        std::string token;
        while (in >> token)
            play(token);
    }
};

void check_restoration(Fixture& f, Move m) {
    const auto before = f.pos.fen(false, true);
    const auto key = f.pos.key();
    const auto pawnKey = f.pos.pawn_key();
    const auto material = f.pos.material_key();
    const auto score = f.pos.psq_score();
    const auto legal = f.moves();
    StateInfo child{};
    f.pos.do_move(m, child);
    ++transitions;
    require(f.pos.pos_is_ok(), "position consistency after make");
    const auto childFen = f.pos.fen(false, true);
    Position restored;
    StateInfo restoredState{};
    restored.set(f.variant, childFen, f.pos.is_chess960(), &restoredState, Threads.main());
    require(restored.fen(false, true) == childFen, "child FEN round trip");
    require(restored.key() == f.pos.key(), "incremental/rebuilt position hash differs: " + childFen);
    require(restored.pawn_key() == f.pos.pawn_key(), "incremental/rebuilt pawn hash differs");
    require(restored.material_key() == f.pos.material_key(), "incremental/rebuilt material hash differs");
    f.pos.undo_move(m);
    require(f.pos.fen(false, true) == before, "make/unmake FEN differs");
    require(f.pos.key() == key && f.pos.pawn_key() == pawnKey && f.pos.material_key() == material,
            "make/unmake hash differs");
    require(f.pos.psq_score() == score, "make/unmake piece-square score differs");
    require(f.moves() == legal, "make/unmake legal move set differs");
}

void all_native_invariants() {
    // Every registered native variant is exercised. This is bounded structural
    // coverage, not a claim that every rule or endgame has been certified.
    for (const std::string& name : variants.get_keys()) {
        run("state-invariants/" + name, [&]() {
            Fixture f(name);
            require(f.pos.pos_is_ok(), "initial position consistency");
            const auto initial = f.pos.fen(false, true);
            Position reload;
            StateInfo state{};
            reload.set(f.variant, initial, f.pos.is_chess960(), &state, Threads.main());
            require(reload.fen(false, true) == initial, "initial FEN round trip");
            require(reload.key() == f.pos.key(), "initial FEN hash round trip");
            std::vector<Move> root;
            for (Move m : MoveList<LEGAL>(f.pos))
                root.push_back(m);
            require(!root.empty(), "initial position has no legal move");
            // Sample the full root list evenly, including late drops/compound moves.
            const std::size_t samples = std::min<std::size_t>(root.size(), 16);
            for (std::size_t i = 0; i < samples; ++i)
                check_restoration(f, root[i * root.size() / samples]);
            std::uint32_t rng = 0x56415231u;
            for (int ply = 0; ply < 24; ++ply) {
                Value result;
                if (f.pos.is_game_end(result))
                    break;
                std::vector<Move> choices;
                for (Move m : MoveList<LEGAL>(f.pos))
                    choices.push_back(m);
                if (choices.empty())
                    break;
                rng = rng * 1664525u + 1013904223u;
                Move m = choices[rng % choices.size()];
                check_restoration(f, m);
                f.states.emplace_back();
                f.pos.do_move(m, f.states.back());
            }
        });
    }
}
} // namespace

int main() {
    pieceMap.init();
    variants.init();
    UCI::init(Options);
    Options["Use NNUE"] = std::string("false");
    PSQT::init(variants.find("chess")->second);
    Bitboards::init();
    Position::init();
    Bitbases::init();
    Endgames::init();
    Threads.set(1);
    Search::clear();
    CurrentProtocol = UCI_GENERAL;

    run("chess/initial-and-illegal-move", []() {
        Fixture f("chess");
        require(f.moves().size() == 20, "expected 20 initial moves");
        auto invalid = std::string("e2e5");
        require(UCI::to_move(f.pos, invalid) == MOVE_NONE, "illegal pawn move accepted");
    });
    run("chess/en-passant-and-undo", []() {
        Fixture f("chess");
        f.play_sequence("e2e4 g8f6 e4e5 d7d5");
        auto m = f.move("e5d6");
        require(f.pos.capture(m), "en passant must be a capture");
        check_restoration(f, m);
    });
    run("chess/promotion-choice", []() {
        Fixture f("chess", "7k/P7/8/8/8/8/8/4K3 w - - 0 1");
        auto moves = f.moves();
        for (const auto* m : {"a7a8q", "a7a8r", "a7a8b", "a7a8n"})
            require(moves.count(m), std::string("missing promotion ") + m);
        require(!moves.count("a7a8"), "unpromoted pawn on last rank");
    });
    run("chess960/castling-notation", []() {
        Fixture f("chess", "4k3/8/8/8/8/8/8/R1K4R w AH - 0 1", true);
        const auto moves = f.moves();
        require(moves.count("c1a1") && moves.count("c1h1"), "Chess960 king-to-rook notation");
        check_restoration(f, f.move("c1a1"));
        check_restoration(f, f.move("c1h1"));
    });
    run("chess/checkmate", []() {
        Fixture f("chess");
        f.play_sequence("f2f3 e7e5 g2g4 d8h4");
        require(f.moves().empty() && f.pos.checkers(), "Fool's mate not detected");
        require(f.pos.checkmate_value() == -VALUE_MATE, "incorrect checkmate result");
    });
    run("chess/threefold-repetition", []() {
        Fixture f("chess");
        f.play_sequence("g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8");
        Value result = VALUE_NONE;
        require(f.pos.is_optional_game_end(result), "threefold repetition missed");
        require(result == VALUE_DRAW, "repetition not a draw");
    });
    run("three-check/final-check-victory", []() {
        Fixture f("3check", "7k/8/8/8/8/8/R7/K7 w - - 1+3 0 1");
        f.play("a2h2");
        Value result = VALUE_NONE;
        require(f.pos.is_immediate_game_end(result) && result == -VALUE_MATE, "third-check victory missing");
    });
    run("king-of-the-hill/center-victory", []() {
        Fixture f("kingofthehill", "7k/8/8/8/8/4K3/8/8 w - - 0 1");
        f.play("e3e4");
        Value result = VALUE_NONE;
        require(f.pos.is_immediate_game_end(result) && result == -VALUE_MATE, "central king victory missing");
    });
    run("crazyhouse/pocket-and-drop-ranks", []() {
        Fixture f("crazyhouse", "4k3/8/8/3n4/4P3/8/8/4K3[P] w - - 0 1");
        auto moves = f.moves();
        require(moves.count("P@d4"), "legal pawn drop missing");
        require(!moves.count("P@a1") && !moves.count("P@a8"), "illegal last-rank pawn drop");
        check_restoration(f, f.move("P@d4"));
        f.play("e4d5");
        require(f.pos.count_in_hand(WHITE, KNIGHT) == 1, "captured knight not in pocket");
    });
    run("antichess/mandatory-capture", []() {
        Fixture f("antichess", "7k/8/8/3p4/4P3/8/8/K7 w - - 0 1");
        require(f.moves() == std::set<std::string>{"e4d5"}, "quiet move allowed despite capture");
    });
    run("atomic/adjacent-kings-and-mate", []() {
        Fixture f("atomic", "8/8/kK6/8/8/8/Q7/8 b - - 0 1");
        require(!f.pos.checkers(), "atomic adjacent kings incorrectly checking");
        Fixture mate("atomic", "BQ6/Rk6/8/8/8/8/8/4K3 b - - 0 1");
        require(mate.moves().empty(), "atomic mate missing");
        // Atomic kings are pseudo-royal in this core. The no-legal-moves
        // adjudicator checks their attacked status through stalemate_value.
        const auto result = mate.pos.checkers() ? mate.pos.checkmate_value() : mate.pos.stalemate_value();
        require(result == -VALUE_MATE, "atomic mate result");
    });
    run("duck/compound-turn", []() {
        Fixture f("duck");
        require(f.moves().size() == 640, "expected 20 piece moves x 32 duck placements");
        const auto m = *MoveList<LEGAL>(f.pos).begin();
        check_restoration(f, m);
    });
    run("seirawan/gate-rights-hash-distinction", []() {
        Fixture withGate("seirawan", "4k3/8/8/8/8/8/8/1N2K3[H] w B - 0 1");
        const auto withKey = withGate.pos.key();
        const auto withMoves = withGate.moves();
        Fixture withoutGate("seirawan", "4k3/8/8/8/8/8/8/1N2K3[H] w - - 0 1");
        require(withMoves != withoutGate.moves(), "gate-right fixture must change legal moves");
        require(withKey != withoutGate.pos.key(), "different gating rights collide in position hash");
    });
    run("ataxx/free-drop-speculative-hash", []() {
        Fixture f("ataxx");
        bool checkedDrop = false;
        for (Move m : MoveList<LEGAL>(f.pos)) {
            if (type_of(m) != DROP)
                continue;
            // Initial clones touch no enemy pieces, so the speculative shortcut
            // must equal the real hash. Elsewhere key_after intentionally does
            // not model every special rule; no broad equivalence is asserted.
            const Key predicted = f.pos.key_after(m);
            StateInfo child{};
            f.pos.do_move(m, child);
            require(predicted == f.pos.key(), "free-drop prefetch hash consumes nonexistent pocket piece");
            f.pos.undo_move(m);
            checkedDrop = true;
        }
        require(checkedDrop, "Ataxx fixture did not exercise a free-drop move");
    });
    run("bughouse/virtual-drop-signed-pocket-hash", []() {
        // These checking drops are used inside partner-board search, although
        // the public LEGAL move list deliberately excludes borrowed pieces.
        // Internal debt cannot be represented by ordinary FEN: do not serialize
        // these states or pretend a partner has actually supplied the pieces.
        const std::vector<Move> sequence{
            make_drop(SQ_F6, KNIGHT, KNIGHT), make_move(SQ_E8, SQ_F8),
            make_drop(SQ_E7, PAWN, PAWN), make_move(SQ_F8, SQ_G7),
            make_drop(SQ_H6, PAWN, PAWN)
        };
        const auto exercise = [&](const std::string& pocket, int finalPawnDebt) {
            Fixture f("bughouse", "4k3/8/8/8/8/8/8/4K3[" + pocket + "] w - - 0 1");
            struct Snapshot { Key key, pawn, material; Score score; int pawns, knights; };
            std::vector<Snapshot> snapshots;
            for (Move m : sequence) {
                require(f.pos.pseudo_legal(m) && f.pos.legal(m), "invalid virtual-drop regression sequence");
                if (type_of(m) == DROP)
                    require(f.pos.gives_check(m), "borrowed piece must give check in this fixture");
                snapshots.push_back({f.pos.key(), f.pos.pawn_key(), f.pos.material_key(),
                                     f.pos.psq_score(), f.pos.count_in_hand(WHITE, PAWN),
                                     f.pos.count_in_hand(WHITE, KNIGHT)});
                const Key predicted = f.pos.key_after(m);
                f.states.emplace_back();
                f.pos.do_move(m, f.states.back());
                require(f.pos.key() == predicted, "signed-pocket prefetch key differs from actual next key");
            }
            require(f.pos.count_in_hand(WHITE, KNIGHT) == -1, "virtual knight debt not exercised");
            require(f.pos.count_in_hand(WHITE, PAWN) == finalPawnDebt, "virtual pawn debt not exercised");
            const std::array<Key, 3> finalKeys{f.pos.key(), f.pos.pawn_key(), f.pos.material_key()};
            for (std::size_t i = sequence.size(); i-- > 0;) {
                f.pos.undo_move(sequence[i]);
                f.states.pop_back();
                const auto& before = snapshots[i];
                require(f.pos.key() == before.key && f.pos.pawn_key() == before.pawn
                        && f.pos.material_key() == before.material, "virtual-drop undo hash mismatch");
                require(f.pos.psq_score() == before.score, "virtual-drop undo score mismatch");
                require(f.pos.count_in_hand(WHITE, PAWN) == before.pawns
                        && f.pos.count_in_hand(WHITE, KNIGHT) == before.knights,
                        "virtual-drop undo reserve mismatch");
            }
            return finalKeys;
        };
        const auto debtTwo = exercise("", -2);
        const auto debtOne = exercise("P", -1);
        require(debtTwo[1] == debtOne[1] && debtTwo[2] == debtOne[2],
                "signed-reserve comparison must have the same on-board pawn/material state");
        require(debtTwo[0] != debtOne[0], "different negative reserve debts collide in position hash");
    });
    run("shogi/nifu-and-drop-rank", []() {
        Fixture f("shogi", "4k4/9/9/9/4P4/9/9/9/4K4[P] w - - 0 1");
        const auto moves = f.moves();
        require(moves.count("P@d5"), "legal shogi pawn drop missing");
        require(!moves.count("P@e4") && !moves.count("P@e6"), "nifu accepted");
        require(!moves.count("P@d9"), "immobile pawn drop accepted");
        check_restoration(f, f.move("P@d5"));
    });
    run("shogi/forced-promotion", []() {
        Fixture f("shogi", "8k/3P5/9/9/9/9/9/9/4K4[] w - - 0 1");
        require(f.moves().count("d8d9+"), "forced promotion missing");
        require(!f.moves().count("d8d9"), "unpromoted immobile pawn accepted");
        check_restoration(f, f.move("d8d9+"));
    });
    run("shogi/optional-promotion", []() {
        Fixture f("shogi", "8k/9/9/3P5/9/9/9/9/4K4[] w - - 0 1");
        require(f.moves().count("d6d7+") && f.moves().count("d6d7"), "promotion option missing");
    });
    run("shogi/sfen-round-trip", []() {
        Fixture f("shogi");
        const auto sfen = f.pos.fen(true);
        Position reload;
        StateInfo state{};
        reload.set(f.variant, sfen, false, &state, Threads.main(), true);
        require(reload.fen(true) == sfen, "SFEN round trip differs");
        require(reload.key() == f.pos.key(), "SFEN round trip hash differs");
    });
    run("shogi/fourfold-repetition", []() {
        Fixture f("shogi");
        for (int i = 0; i < 3; ++i)
            f.play_sequence("e1e2 e9e8 e2e1 e8e9");
        Value result = VALUE_NONE;
        require(f.pos.is_optional_game_end(result) && result == VALUE_DRAW, "fourfold shogi repetition missing");
    });
    run("shogi/upstream-uchifuzume-loss-adjudication", []() {
        Fixture f("shogi", "lnsg3nk/1r2b1gs1/ppppppp1p/7N1/7p1/9/PPPPPPPP1/1B5R1/LNSGKGS1L[P] w 0 1");
        // The inherited FSF contract admits the move, then penalizes the
        // dropper. This test deliberately does not claim strict legal filtering.
        f.play("P@i8");
        require(f.moves().empty() && f.pos.checkers(), "pawn-drop mate fixture");
        require(f.pos.checkmate_value() == VALUE_MATE, "pawn-drop mate must lose for dropper");
    });
    run("janggi/pass-in-stalemate", []() {
        Fixture f("janggi", "4k4/c7R/9/3R1R3/9/9/9/9/9/3K5 b - - 0 1");
        require(f.moves() == std::set<std::string>{"e10e10"}, "janggi pass missing");
        check_restoration(f, f.move("e10e10"));
    });
    run("janggi/palace-discovered-check", []() {
        Fixture f("janggi", "4ka3/4a4/9/4R4/2B6/9/9/5K3/4p4/3r5 b - - 0 113");
        f.play("e2f2");
        require(bool(f.pos.checkers()), "palace diagonal discovered check missed");
    });
    run("janggi/cannon-screen-and-capture", []() {
        Fixture f("janggi", "1Pbcka3/3nNn1c1/N2CaC3/1pB6/9/9/5P3/9/4K4/9 w - - 0 23");
        require(f.moves().count("f8f10"), "legal screened cannon capture missing");
        check_restoration(f, f.move("f8f10"));
    });
    run("janggi/bikjang-traditional-and-points", []() {
        for (const auto* name : {"janggi", "janggitraditional"}) {
            Fixture f(name, "4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1");
            f.play("e1e1");
            Value result = VALUE_NONE;
            require(f.pos.is_immediate_game_end(result), "accepted bikjang did not end game");
            require(result == (std::string(name) == "janggi" ? VALUE_MATE : VALUE_DRAW),
                    "bikjang point/draw mode distinction missing");
        }
    });
    run("xiangqi/screened-cannon", []() {
        Fixture f("xiangqi");
        require(f.moves().count("h3h10"), "screened cannon capture missing");
        f.play("h3h10");
        require(f.moves().count("i10h10"), "rook recapture missing");
    });
    run("xiangqi/repeated-cannon-chase", []() {
        Fixture f("xiangqi", "2bakabnr/9/r1n1c4/2p1p1p1p/PP7/9/4P1P1P/2C3NC1/9/1NBAKAB1R w - - 0 1");
        f.play_sequence("c3a3 a8b8 a3b3 b8a8 b3a3 a8b8 a3b3 b8a8 b3a3");
        Value result = VALUE_NONE;
        require(f.pos.is_optional_game_end(result), "repeated chase missed");
        require(result == VALUE_MATE, "chaser adjudication sign differs from upstream fixture");
    });
    run("makruk/promotion-region", []() {
        Fixture f("makruk", "7k/8/8/3P4/8/8/8/4K3 w - - 0 1");
        require(f.moves().count("d5d6m"), "sixth-rank makruk promotion missing");
        require(!f.moves().count("d5d6"), "unpromoted makruk pawn accepted");
        check_restoration(f, f.move("d5d6m"));
    });
    all_native_invariants();

    Threads.set(0);
    variants.clear_all();
    pieceMap.clear_all();
    std::cout << "RULE_TEST_RESULT {\"cases\":" << cases << ",\"checks\":" << checks
              << ",\"transitions\":" << transitions << ",\"failures\":" << failures << "}\n";
    return failures ? 1 : 0;
}
