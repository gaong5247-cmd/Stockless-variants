#!/usr/bin/env python3
"""Reject resealed mislabeled data and verify independent game/input holdouts.

Default fixtures exercise the dataset contract, not game legality. --data can
additionally exercise the same regressions on a fresh native-generator output.
Changing the file hashes after every mutation ensures these tests reach semantic
validation instead of succeeding merely because a checksum is stale.

Copyright (C) 2026 Stockless Variants contributors
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/nnue"))
from model import digest
from train import load_dataset, partition
from validate import checked_split

EXTERNAL_DATA: Path | None = None


def fixture() -> tuple[dict, list[dict], list[dict]]:
    """Small schema fixtures, deliberately independent of an installed engine."""
    rule_hash = hashlib.sha256(b"schema fixture rules").hexdigest()
    engine_hash = hashlib.sha256(b"schema fixture teacher").hexdigest()
    descriptor = {"supported": True, "format_version": 1, "feature_version": 1,
                  "variant": "chess", "rules_sha256": rule_hash, "width": 8, "height": 8,
                  "type_ids": [1, 2, 3, 4, 5, 63], "input_dim": 780,
                  "hidden_dim": 64, "qa": 256, "qb": 64}
    provenance = {"variant": "chess", "rules_sha256": rule_hash, "teacher_engine_sha256": engine_hash}
    claim = "claim_all_native_optional"
    games, rows = [], []
    for game_index in range(6):
        game_id = hashlib.sha256(f"fixture game {game_index}".encode()).hexdigest()
        final = {"result_white": None, "terminal_reason": "max_plies", "truncated": True}
        games.append({
            "schema": 1, **provenance, "game_id": game_id, "game_index": game_index,
            "moves": ["a2a3"] * 16, "plies": 16, "opening_plies": 0, "rows": 12,
            "final_side_to_move": 0, "claim_policy": claim, **final,
            "native_terminal": {"ended": False, "result_stm": None, "reason": None, "claim_policy": claim},
        })
        for ply in range(12):
            raw = 50 + 10 * game_index + ply
            rows.append({
                "schema": 1, **provenance, "game_id": game_id, "ply": ply,
                "side_to_move": ply % 2, "features": [[game_index * 65 + ply, 1]],
                "label_cp_white": float(raw if ply % 2 == 0 else -raw), **final,
                "teacher": {"unit": "cp", "value": raw, "perspective": "side_to_move",
                            "bound": None, "depth": 2, "nodes": 50},
            })
    manifest = {"schema": 1, "variant": "chess", "descriptor": descriptor,
                "rules_sha256": rule_hash, "engine_sha256": engine_hash,
                "limits": {"max_plies": 16}, "game_count": len(games), "rows": len(rows),
                "completed_games": 0, "truncated_games": len(games),
                "teacher": {"engine_sha256": engine_hash, "score_perspective": "side_to_move",
                            "label_perspective": "white", "label_cp_limit": 10000, "claim_policy": claim}}
    return manifest, rows, games


def read_external(directory: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest, rows, games = load_dataset(directory)
    return manifest, rows, games


def reseal(directory: Path, manifest: dict, rows: list[dict], games: list[dict]) -> None:
    """Update only file checksums/sizes; preserve semantic claims under test."""
    manifest = copy.deepcopy(manifest)
    files = {}
    for name, records in (("positions.jsonl", rows), ("games.jsonl", games)):
        path = directory / name
        path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
        files[name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    manifest["files"] = files
    (directory / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")


class DatasetIntegrity(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="svnnue-data-test-")
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.manifest, self.rows, self.games = copy.deepcopy(read_external(EXTERNAL_DATA) if EXTERNAL_DATA else fixture())

    def load(self) -> tuple[dict, list[dict], list[dict]]:
        reseal(self.directory, self.manifest, self.rows, self.games)
        return load_dataset(self.directory)

    def first_cp(self, side: int | None = None) -> dict:
        return next(row for row in self.rows if row["teacher"]["unit"] == "cp"
                    and (side is None or row["side_to_move"] == side))

    def write_split(self, split: dict) -> str:
        path = self.directory / "split.json"
        path.write_text(json.dumps(split, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        return digest(path)

    def sealed_model_split(self) -> tuple[list[dict], list[dict], dict, dict, dict]:
        # No network training is needed to test the binding of heldout rows.
        # Use the same public partition routine that records a trained model's
        # frozen split; load_dataset supplies verified input identities.
        _, rows, games = self.load()
        _, split = partition(rows, games, 77123)
        manifest = {"training_seed": split["seed"], "split_sha256": self.write_split(split)}
        training = {"seed": split["seed"],
                    "split": {key: value for key, value in split.items() if key != "position_indices"}}
        return rows, games, split, manifest, training

    def test_unchanged_dataset_accepted(self) -> None:
        manifest, rows, games = self.load()
        self.assertEqual(len(rows), manifest["rows"])
        self.assertEqual(len(games), manifest["game_count"])

    def test_resealed_cp_label_change_rejected(self) -> None:
        row = self.first_cp()
        row["label_cp_white"] += -1 if row["label_cp_white"] >= 10000 else 1
        with self.assertRaisesRegex(ValueError, "raw side-to-move teacher score"):
            self.load()

    def test_black_cp_perspective_inversion_rejected(self) -> None:
        row = self.first_cp(side=1)
        row["teacher"]["value"] = 123
        row["label_cp_white"] = 123.0
        with self.assertRaisesRegex(ValueError, "raw side-to-move teacher score"):
            self.load()

    def test_mate_target_is_explicit_and_checked(self) -> None:
        row = self.first_cp(side=1)
        row["teacher"].update(unit="mate", value=3)
        row["label_cp_white"] = -10000.0
        self.load()
        row["label_cp_white"] = 10000.0
        with self.assertRaisesRegex(ValueError, "raw side-to-move teacher score"):
            self.load()

    def test_boolean_label_rejected(self) -> None:
        row = self.first_cp(side=0)
        row["teacher"]["value"] = 1
        row["label_cp_white"] = True
        with self.assertRaisesRegex(ValueError, "Invalid teacher label"):
            self.load()

    def test_missing_or_invalid_raw_score_rejected(self) -> None:
        row = self.first_cp()
        original = copy.deepcopy(row["teacher"])
        mutations = [{"bound": None}, {**original, "value": True}, {**original, "depth": 0},
                     {**original, "nodes": False}, {**original, "perspective": "white"}]
        for invalid in mutations:
            with self.subTest(score=invalid):
                row["teacher"] = invalid
                with self.assertRaisesRegex(ValueError, "raw teacher score metadata"):
                    self.load()

    def test_bound_is_not_an_exact_teacher_label(self) -> None:
        self.first_cp()["teacher"]["bound"] = "lowerbound"
        with self.assertRaisesRegex(ValueError, "search bound"):
            self.load()

    def test_resealed_truncated_draw_rejected(self) -> None:
        game = next(game for game in self.games if game["truncated"])
        game["result_white"] = 0
        for row in self.rows:
            if row["game_id"] == game["game_id"]:
                row["result_white"] = 0
        with self.assertRaisesRegex(ValueError, "Truncated game must retain null result"):
            self.load()

    def test_position_truncation_must_match_game(self) -> None:
        self.rows[0]["truncated"] = not self.rows[0]["truncated"]
        with self.assertRaisesRegex(ValueError, "Position/game result, truncation"):
            self.load()

    def test_boolean_truncation_required(self) -> None:
        self.rows[0]["truncated"] = int(self.rows[0]["truncated"])
        with self.assertRaisesRegex(ValueError, "Position/game result, truncation"):
            self.load()

    def test_resealed_invalid_result_rejected(self) -> None:
        game = self.games[0]
        game["result_white"] = 42
        for row in self.rows:
            if row["game_id"] == game["game_id"]:
                row["result_white"] = 42
        with self.assertRaisesRegex(ValueError, "Invalid game result"):
            self.load()

    def test_completed_result_preserves_native_side(self) -> None:
        # A result from a black-to-move terminal position must be inverted when
        # expressed as a white-perspective game outcome.
        game = self.games[0]
        was_truncated = game["truncated"]
        game.update(truncated=False, result_white=-1, terminal_reason="checkmate", final_side_to_move=1)
        game["native_terminal"].update(ended=True, result_stm=1, reason="checkmate")
        if was_truncated:
            self.manifest["truncated_games"] -= 1
            self.manifest["completed_games"] += 1
        for row in self.rows:
            if row["game_id"] == game["game_id"]:
                row.update(truncated=False, result_white=-1, terminal_reason="checkmate")
        self.load()
        game["native_terminal"]["result_stm"] = -1
        with self.assertRaisesRegex(ValueError, "native signed terminal result"):
            self.load()

    def test_position_ply_must_precede_game_end(self) -> None:
        row = self.rows[0]
        game = next(game for game in self.games if game["game_id"] == row["game_id"])
        row["ply"] = game["plies"]
        with self.assertRaisesRegex(ValueError, "out-of-range game position ply"):
            self.load()

    def test_game_plies_must_match_history(self) -> None:
        self.games[0]["moves"].pop()
        with self.assertRaisesRegex(ValueError, "game plies, moves"):
            self.load()

    def test_resealed_position_provenance_mix_rejected(self) -> None:
        # Chess and Makruk have the same D=780. Dimensions alone cannot detect
        # an accidental concatenation; each record must carry its source IDs.
        row = self.rows[0]
        for key, value in (("variant", "makruk" if row["variant"] != "makruk" else "chess"),
                           ("rules_sha256", "0" * 64), ("teacher_engine_sha256", "0" * 64)):
            with self.subTest(key=key):
                original = row[key]
                row[key] = value
                with self.assertRaisesRegex(ValueError, "Position variant/rules/teacher provenance"):
                    self.load()
                row[key] = original

    def test_resealed_game_provenance_mix_rejected(self) -> None:
        self.games[0]["variant"] = "makruk" if self.games[0]["variant"] != "makruk" else "chess"
        with self.assertRaisesRegex(ValueError, "Game variant/rules/teacher provenance"):
            self.load()

    def test_split_excludes_shared_games_and_identical_inputs(self) -> None:
        # Deliberately repeat a model input across every game. The split must
        # eliminate this overlap even though all six game IDs are distinct.
        for game in self.games:
            candidates = [row for row in self.rows if row["game_id"] == game["game_id"] and row["side_to_move"] == 0]
            if candidates:
                candidates[0]["features"] = [[777, 1]]
        _, rows, games = self.load()
        parts, report = partition(rows, games, 77123)
        repeated, again = partition(rows, games, 77123)
        names = ("train", "validation", "test")
        for name in names:
            self.assertEqual(parts[name].tolist(), repeated[name].tolist())
        self.assertEqual(report, again)
        game_sets = {name: {rows[int(i)]["game_id"] for i in parts[name]} for name in names}
        input_sets = {name: {rows[int(i)]["input_sha256"] for i in parts[name]} for name in names}
        for i, name in enumerate(names):
            for other in names[i + 1:]:
                self.assertFalse(game_sets[name] & game_sets[other])
                self.assertFalse(input_sets[name] & input_sets[other])
        retained = [rows[int(i)] for name in names for i in parts[name]]
        self.assertEqual(sum(row["side_to_move"] == 0 and row["features"] == [[777, 1]] for row in retained), 1)
        self.assertGreater(sum(report["removed_duplicate_inputs"].values()), 0)

    def test_sealed_model_split_accepted(self) -> None:
        rows, games, split, manifest, training = self.sealed_model_split()
        self.assertEqual(checked_split(self.directory, manifest, training, rows, games), split)

    def test_changed_model_split_checksum_rejected(self) -> None:
        rows, games, split, manifest, training = self.sealed_model_split()
        changed = copy.deepcopy(split)
        changed["position_indices"]["test"][0] = changed["position_indices"]["train"][0]
        self.write_split(changed)  # Keep the manifest's original checksum.
        with self.assertRaisesRegex(ValueError, "split checksum"):
            checked_split(self.directory, manifest, training, rows, games)

    def test_resealed_model_split_train_test_swaps_rejected(self) -> None:
        rows, games, split, manifest, training = self.sealed_model_split()
        with self.subTest(mutation="one row moved between train and test"):
            changed = copy.deepcopy(split)
            groups = changed["position_indices"]
            groups["train"][0], groups["test"][0] = groups["test"][0], groups["train"][0]
            resealed_manifest = {**manifest, "split_sha256": self.write_split(changed)}
            with self.assertRaisesRegex(ValueError, "heldout game membership"):
                checked_split(self.directory, resealed_manifest, training, rows, games)

        with self.subTest(mutation="entire train and test memberships swapped and resealed"):
            changed = copy.deepcopy(split)
            # This forgery has disjoint rows/inputs, consistent game ownership,
            # matching partition counts, and an updated metrics summary. Only
            # reconstruction from the recorded seed proves these were not the
            # test inputs chosen before training.
            for key in ("position_indices", "game_ids", "retained_positions", "removed_duplicate_inputs"):
                groups = changed[key]
                groups["train"], groups["test"] = groups["test"], groups["train"]
            resealed_manifest = {**manifest, "split_sha256": self.write_split(changed)}
            forged_training = {"seed": training["seed"],
                               "split": {key: value for key, value in changed.items()
                                         if key != "position_indices"}}
            with self.assertRaisesRegex(ValueError, "seeded whole-game partition"):
                checked_split(self.directory, resealed_manifest, forged_training, rows, games)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="also use a fresh, strict-provenance native dataset")
    args, remaining = parser.parse_known_args()
    EXTERNAL_DATA = args.data
    unittest.main(argv=[sys.argv[0], *remaining])
