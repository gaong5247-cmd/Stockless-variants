// SPDX-License-Identifier: AGPL-3.0-or-later
// Boundary and variant-safety tests for the Stockless -> FSF search adapter.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "bitboard.h"
#include "endgame.h"
#include "piece.h"
#include "position.h"
#include "psqt.h"
#include "search.h"
#include "thread.h"
#include "uci.h"
#include "variant.h"
#include "stockless_core/search_policy.h"
#include "stockless_core/variant_policy.h"

using namespace Stockfish;
using namespace Stockfish::Stockless;

namespace {
unsigned checks = 0;
void require(bool ok, const std::string& why) {
    ++checks;
    if (!ok)
        throw std::runtime_error(why);
}

struct Fixture {
    StateInfo state{};
    Position pos;
    explicit Fixture(const std::string& name) {
        const auto it = variants.find(name);
        require(it != variants.end(), "missing variant " + name);
        UCI::init_variant(it->second);
        PSQT::init(it->second);
        pos.set(it->second, it->second->startFen, it->second->chess960, &state, Threads.main());
    }
};
}

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
    bool passed = true;
    try {
        // Behavior/property oracle: bounded, deterministic, monotone fractional
        // quantization. No copy of the adapter's bit-mixing implementation.
        for (std::uint64_t selector = 0; selector < 4096; ++selector)
            for (int native = -4; native <= 16; ++native) {
                require(adapted_reduction(native, 0, selector) == native, "hybrid-disabled reduction changed");
                int previous = native;
                for (int adjustment : {-1, -319, -320, -511, -512, -513, -640, -831, -832}) {
                    const int actual = adapted_reduction(native, adjustment, selector);
                    require(actual == native || actual == native - 1, "buyback exceeded one ply");
                    require(actual <= previous, "larger tactical budget increased reduction");
                    require(adapted_reduction(native, adjustment, selector) == actual, "selector not deterministic");
                    previous = actual;
                }
                require(adapted_reduction(native, 512, selector) == native, "positive adjustment was not clamped");
                require(adapted_reduction(native, -2048, selector) == previous, "negative adjustment exceeded cap");
            }
        // Including the mobile cap proves that small sub-ply signals survive.
        for (int budget : {320, 511, 512, 513, 640, 832}) {
            int changed = 0;
            for (std::uint64_t selector = 0; selector < 65536; ++selector)
                changed += adapted_reduction(2, -budget, selector) == 1;
            const double fraction = changed / 65536.0;
            require(std::abs(fraction - budget / 1024.0) < 0.02,
                    "fractional budget not reflected across stable selectors");
        }

        const SearchSettings desktop{true, true, true, false, 145};
        for (const auto* name : {"antichess", "giveaway", "horde", "duck", "bughouse", "kingofthehill", "racingkings"}) {
            Fixture f(name);
            const auto profile = variant_search_policy(f.pos);
            const auto setting = variant_search_settings(f.pos, desktop);
            require(!profile.hybrid && !profile.nullMove, std::string("unsafe search objective enabled: ") + name);
            require(!setting.hybrid && !setting.threats && !setting.overdrive,
                    std::string("unsafe hybrid setting leaked: ") + name);
        }
        for (const auto* name : {"atomic", "shogi", "janggi", "xiangqi", "makruk", "crazyhouse", "capablanca", "3check"}) {
            Fixture f(name);
            require(variant_search_policy(f.pos).hybrid, std::string("regional hybrid unexpectedly disabled: ") + name);
        }
        for (const auto* name : {"shogi", "janggi", "crazyhouse", "3check"}) {
            Fixture f(name);
            const auto profile = variant_search_policy(f.pos);
            require(!profile.overdrive && !profile.nullMove, std::string("conservative regional guard missing: ") + name);
        }
        {
            Fixture f("chess");
            const auto profile = variant_search_policy(f.pos);
            require(profile.hybrid && profile.overdrive && profile.nullMove, "orthodox profile unexpectedly disabled");
        }
        {
            Fixture f("atomic");
            const auto profile = variant_search_policy(f.pos);
            require(!profile.overdrive && !profile.nullMove, "atomic aggressive extension/null-move guard missing");
        }
        const SearchNodeInputs forcing{256, false, true, true, false, {256, 256, 256}};
        for (int aggression : {0, 100, 145, 200}) {
            SearchSettings mobile{true, true, true, true, aggression};
            auto prepared = prepare_search_policy(forcing, mobile);
            auto policy = make_prepared_move_policy(prepared, 12, 1, 4, 4, true, true, true);
            require(policy.lmrAdjustment >= -320 && policy.lmrAdjustment <= 0, "mobile cap exceeded");
            require(policy.pruningDepthBoost == 0 && policy.reSearchDepthBonus == 0, "mobile Overdrive leaked");
            mobile.overdrive = false;
            auto off = make_prepared_move_policy(prepare_search_policy(forcing, mobile), 12, 1, 4, 4, true, true, true);
            require(policy.lmrAdjustment == off.lmrAdjustment, "mobile Overdrive changed buyback");
        }

        for (const auto& name : variants.get_keys()) {
            Fixture f(name);
            auto settings = variant_search_settings(f.pos, desktop);
            const auto signal = compute_threat_signal(f.pos, 8, true, settings);
            require(signal.attackPressure >= 0 && signal.attackPressure <= 256, name + ": attack bound");
            require(signal.kingDanger >= 0 && signal.kingDanger <= 256, name + ": danger bound");
            require(signal.tacticalVolatility >= 0 && signal.tacticalVolatility <= 256, name + ": volatility bound");
            settings.hybrid = false;
            const auto disabled = compute_threat_signal(f.pos, 8, true, settings);
            require(disabled.attackPressure == 0 && disabled.kingDanger == 0 && disabled.tacticalVolatility == 0,
                    name + ": disabled threat signal not zero");
        }
    } catch (const std::exception& error) {
        passed = false;
        std::cerr << "FAIL policy: " << error.what() << '\n';
    }
    Threads.set(0);
    variants.clear_all();
    pieceMap.clear_all();
    std::cout << "POLICY_TEST_RESULT {\"checks\":" << checks
              << ",\"failures\":" << (passed ? 0 : 1) << "}\n";
    return passed ? 0 : 1;
}
