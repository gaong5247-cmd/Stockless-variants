#!/usr/bin/env python3
"""Negative-input and transaction tests against the compiled engine.

SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile

from run_engine_tests import Engine, ROOT, sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--variants", type=Path, default=ROOT / "upstream/fairy-stockfish/src/variants.ini")
    args = parser.parse_args()
    engine = args.engine.resolve()
    results: list[dict] = []

    def record(name: str) -> None:
        results.append({"test": name, "passed": True})

    def checked(path: Path, valid: bool) -> subprocess.CompletedProcess:
        run = subprocess.run([str(engine), "check", str(path)], text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        assert (run.returncode == 0) == valid, (path.name, run.returncode, run.stdout)
        if not valid:
            assert "ERROR" in run.stdout, (path.name, run.stdout)
        return run

    checked(args.variants.resolve(), True)
    record("pinned_variants_ini_accepts_supported_syntax")
    with tempfile.TemporaryDirectory(prefix="stockless-protocol-") as temp:
        directory = Path(temp)
        good = directory / "good.ini"
        good.write_text("[alias_test:chess]\ncastling = true\ncastling = false\n", encoding="utf-8")
        checked(good, True)
        record("repeated_ini_keys_keep_last_assignment_semantics")
        negatives = {
            "unknown_attribute": "[invalid:chess]\nunknownRule = true\n",
            "unknown_template": "[invalid:missing_template]\ncastling = false\n",
            "invalid_boolean": "[invalid:chess]\nchecking = maybe\n",
            "invalid_number": "[invalid:chess]\nmaxRank = 9999999999999999999999\n",
            "invalid_board_size": "[invalid:chess]\nmaxFile = 99\n",
            "invalid_region": "[invalid:chess]\npromotionRegionWhite = *oops\n",
            "negative_region": "[invalid:chess]\npromotionRegionWhite = *-1\n",
            "unsupported_betza": "[invalid:chess]\ncustomPiece1 = x:Qz\n",
            "missing_section": "castling = false\n",
            "missing_value": "[invalid:chess]\ncastling =\n",
            "invalid_fen": "[invalid:chess]\nstartFen = 8/8/8/8/8/8/8/7 w - - 0 1\n",
            "incompatible_rules": "[invalid:crazyhouse]\nwallingRule = duck\n",
        }
        for name, content in negatives.items():
            path = directory / f"{name}.ini"
            path.write_text(content, encoding="utf-8")
            checked(path, False)
            record(name)
        checked(directory / "does_not_exist.ini", False)
        record("missing_file_fails")

        transaction = directory / "transaction.ini"
        transaction.write_text("[staged_good:chess]\ncastling = false\n"
                               "[staged_bad:chess]\nunknownRule = true\n", encoding="utf-8")
        run = subprocess.run([str(engine), "noautoload"],
                             input=f"uci\nload {transaction}\nuci\nquit\n", text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        assert run.returncode != 0 and "ERROR" in run.stdout, run.stdout
        assert " var staged_good" not in run.stdout and " var staged_bad" not in run.stdout, run.stdout
        record("invalid_config_does_not_partially_register_variants")

        with Engine(engine) as client:
            client.option("VariantConfigPath", str(good))
            client.option("VariantName", "alias_test")
            client.position()
            assert client.perft(1)[0] == 20
            fen = next(line for line in client.dump() if line.startswith("Fen:"))
            assert fen.split()[3] == "-", fen
            record("variant_name_and_config_path_aliases")

        # These definitions deliberately share FEN and piece IDs but disagree
        # about movement. Switching rules must invalidate TT and evaluation
        # caches even when the GUI sends no ucinewgame command.
        isolation = directory / "cache-isolation.ini"
        isolation.write_text(
            "[cache_knight:chess]\ncustomPiece1 = x:N\ncastling = false\n"
            "pieceToCharTable = -\nstartFen = 4k3/8/8/3x4/3X4/8/8/4K3 w - - 0 1\n"
            "[cache_rook:cache_knight]\ncustomPiece1 = x:R\n", encoding="utf-8")
        checked(isolation, True)

        def search_signature(client: Engine) -> tuple:
            best, lines = client.search("depth 5")
            completed = [line for line in lines if re.search(r"\bdepth 5\b", line)
                         and " score " in line and " pv " in line]
            assert completed, lines
            last = completed[-1]
            score = re.search(r"\bscore (cp|mate) (-?\d+)", last)
            assert score, last
            return best, score.groups(), last.split(" pv ", 1)[1]

        with Engine(engine) as client:
            client.option("VariantConfigPath", str(isolation))
            client.option("VariantName", "cache_knight")
            client.position()
            search_signature(client)
            client.option("UCI_Variant", "cache_rook")
            client.position()
            switched = search_signature(client)
        with Engine(engine) as fresh:
            fresh.option("VariantConfigPath", str(isolation))
            fresh.option("VariantName", "cache_rook")
            fresh.position()
            expected = search_signature(fresh)
        assert switched == expected, (switched, expected)
        record("variant_switch_clears_search_and_evaluation_caches_without_newgame")

    with Engine(engine) as client:
        assert any("option name Use NNUE type check default false" in line for line in client.handshake)
        assert any("option name EvalBackend type combo" in line and "var Auto" in line
                   and "var Classical" in line for line in client.handshake)
        record("classical_default_and_explicit_backend_capabilities")
        client.position("startpos", ["e2e4"])
        before = next(line for line in client.dump() if line.startswith("Fen:"))
        for command, expected in [
            ("position startpos moves e2e4 e7e6 e4e6", "illegal move"),
            ("position fen 8/8/8/8/8/8/8/7 w - - 0 1", "invalid FEN"),
            ("position fen 99999K/8/8/8/8/8/8/k7 w - - 0 1", "invalid FEN"),
        ]:
            client.send(command)
            output = client.ready()
            assert any(expected in line for line in output), output
            after = next(line for line in client.dump() if line.startswith("Fen:"))
            assert after == before, (command, before, after)
        record("invalid_position_and_move_sequences_leave_previous_state_intact")

        for name, value in [("Hash", "NaN"), ("Threads", "junk"),
                            ("StocklessAggression", "201"), ("StocklessHybrid", "maybe"),
                            ("EvalBackend", "ShogiNNUE"), ("Use NNUE", "true")]:
            client.send(f"setoption name {name} value {value}")
            output = client.ready()
            assert any("ERROR" in line for line in output), (name, value, output)
        record("invalid_options_and_unvalidated_nnue_activation_are_rejected")

        client.send("go infinite")
        client.send("setoption name Hash value 4")
        output = client.ready()
        assert any(line.startswith("bestmove ") for line in output), output
        client.position()
        assert client.perft(2)[0] == 400
        record("option_mutation_stops_and_joins_infinite_search")
        client.send("go infinite")
        client.send("position startpos moves d2d4")
        client.ready()
        after = next(line for line in client.dump() if line.startswith("Fen:"))
        assert "3P4" in after, after
        record("position_mutation_stops_and_joins_infinite_search")

    with Engine(engine, protocol="usi") as client:
        client.position("sfen lnsgkgsnl/1r5b1/ppppppppp/9/9/9/PPPPPPPPP/1B5R1/LNSGKGSNL b - 1",
                        ["7g7f"])
        assert client.perft(1)[0] > 0
        before = next(line for line in client.dump() if line.startswith("Fen:"))
        client.send("position sfen 9/9/9/9/9/9/9/9/9 b 99999999999999P 1")
        output = client.ready()
        assert any("invalid SFEN" in line for line in output), output
        assert next(line for line in client.dump() if line.startswith("Fen:")) == before
        record("sfen_validation_and_transaction")

    report = {"schema": 1, "engine": str(engine), "engine_sha256": sha256(engine),
              "passed": len(results), "failed": 0, "results": results}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Protocol validation: {len(results)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
