#!/usr/bin/env python3
"""Validate a trained, quantized model in the actual native search engine.

This gate checks exact feature/integer parity, accumulator consistency, legal
searches and network/data identities. It does not measure playing strength.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from generate import Engine
from model import digest, integer_predict, read_network
from train import load_dataset, partition, save_json


def checked_split(model_dir: Path, manifest: dict, training: dict,
                  rows: list[dict], games: list[dict]) -> dict:
    """Bind the actual heldout indices, not just the reported partition sizes.

    Reconstructing the seeded split also rejects a resealed replacement that
    assigns training rows/games to the test set. Neither reconstruction nor
    validation-position selection consults teacher labels.
    """
    path = model_dir / "split.json"
    if digest(path) != manifest.get("split_sha256"):
        raise ValueError("Model split checksum mismatch or missing split binding")
    split = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(split, dict):
        raise ValueError("Model split must be an object")
    seed = manifest.get("training_seed")
    if (type(seed) is not int or not 0 <= seed < 2 ** 32
            or type(split.get("seed")) is not int or split["seed"] != seed
            or type(training.get("seed")) is not int or training["seed"] != seed):
        raise ValueError("Model, metrics and split training seeds differ")
    names = {"train", "validation", "test"}
    indices = split.get("position_indices")
    game_ids = split.get("game_ids")
    if (not isinstance(indices, dict) or set(indices) != names
            or not isinstance(game_ids, dict) or set(game_ids) != names):
        raise ValueError("Model split must define train, validation and test memberships")
    known_games = {game["game_id"] for game in games}
    seen_games, seen_rows, seen_inputs = set(), set(), set()
    for name in ("train", "validation", "test"):
        group = game_ids[name]
        selected = indices[name]
        if (not isinstance(group, list) or not group
                or any(type(game_id) is not str or game_id not in known_games for game_id in group)
                or len(set(group)) != len(group) or seen_games.intersection(group)):
            raise ValueError("Model split game memberships are invalid or overlap")
        if (not isinstance(selected, list) or not selected
                or any(type(index) is not int or not 0 <= index < len(rows) for index in selected)
                or len(set(selected)) != len(selected) or seen_rows.intersection(selected)):
            raise ValueError("Model split row indices are invalid, duplicated or overlap")
        group_set = set(group)
        for index in selected:
            row = rows[index]
            if row["game_id"] not in group_set:
                raise ValueError("Model split row does not belong to its heldout game membership")
            if row["input_sha256"] in seen_inputs:
                raise ValueError("Model split repeats an identical input across retained rows")
            seen_inputs.add(row["input_sha256"])
        seen_games.update(group)
        seen_rows.update(selected)
    _, reconstructed = partition(rows, games, seed)
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if canonical(split) != canonical(reconstructed):
        raise ValueError("Model split differs from the recorded seeded whole-game partition")
    expected_summary = {key: value for key, value in reconstructed.items() if key != "position_indices"}
    if canonical(training.get("split")) != canonical(expected_summary):
        raise ValueError("Metrics heldout split summary differs from verified split indices")
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=64)
    args = parser.parse_args()
    if not 1 <= args.positions <= 1024:
        parser.error("positions must be 1..1024")
    engine_path = args.engine.resolve()
    dataset, rows, games = load_dataset(args.data)
    descriptor = dataset["descriptor"]
    manifest = json.loads((args.model_dir / "network-manifest.json").read_text())
    if manifest["descriptor"] != descriptor or manifest["dataset_manifest_sha256"] != digest(args.data / "manifest.json"):
        raise ValueError("Model does not describe this exact dataset and native rule layout")
    if Path(manifest["network"]).name != manifest["network"]:
        raise ValueError("Network manifest must reference a local basename")
    network = (args.model_dir / manifest["network"]).resolve()
    if digest(network) != manifest["network_sha256"]:
        raise ValueError("Model manifest hash differs from executable network payload")
    if digest(args.model_dir / "metrics.json") != manifest["metrics_sha256"]:
        raise ValueError("Model metrics checksum mismatch")
    training = json.loads((args.model_dir / "metrics.json").read_text())
    if not training["training_completed"] or not training["quantization_passed"] or training["gradient_steps"] <= 0:
        raise ValueError("Training/quantization gate has not passed")
    split = checked_split(args.model_dir, manifest, training, rows, games)
    # Use frozen test inputs; test labels never enter this deterministic parity
    # comparison or its selection of states. Rows remain identified in reports.
    available = split["position_indices"]["test"]
    selected = [available[i * len(available) // min(len(available), args.positions)]
                for i in range(min(len(available), args.positions))]
    if not selected:
        raise ValueError("No independent test positions")
    q = read_network(network, descriptor)
    x = np.zeros((len(selected), descriptor["input_dim"]), dtype=np.int64)
    sides = np.array([rows[index]["side_to_move"] for index in selected], dtype=np.int64)
    for i, index in enumerate(selected):
        for feature, count in rows[index]["features"]:
            x[i, feature] = count
    expected = integer_predict(q, x, sides)
    games_by_id = {game["game_id"]: game for game in games}
    report = {"schema": 1, "engine": str(engine_path), "engine_sha256": digest(engine_path),
              "network_sha256": digest(network), "variant": descriptor["variant"],
              "split_sha256": manifest["split_sha256"],
              "heldout_binding": "sha256-and-reconstructed-seeded-whole-game-partition",
              "heldout_row_indices": selected,
              "rules_sha256": descriptor["rules_sha256"], "passed": 0, "failed": 0,
              "tests": [], "strength_validated": False, "elo": None}
    started = time.monotonic()
    engine = Engine(engine_path, timeout=60)
    try:
        engine.option("UCI_Variant", descriptor["variant"])
        engine.option("Threads", 1)
        engine.option("Hash", 16)
        engine.position([])
        if engine.features()["descriptor"] != descriptor:
            raise ValueError("Native runtime rules/features differ from trained model")
        engine.option("EvalFile", network)
        engine.option("EvalBackend", "StocklessNNUE")
        engine.send("nnuestatus")
        status = json.loads(engine.until(lambda line: line.startswith("info string NNUE_STATUS "))[-1].split("NNUE_STATUS ", 1)[1])
        if not status["active"] or status["backend"] != "StocklessNNUE":
            raise AssertionError("Runtime silently fell back instead of activating trained NNUE")
        report["runtime_status"] = status
        for i, index in enumerate(selected):
            row = rows[index]
            game = games_by_id[row["game_id"]]
            engine.position(game["moves"][:row["ply"]])
            native = engine.features()
            if native["fen"] != row["fen"] or native["features"] != row["features"] or native["side_to_move"] != row["side_to_move"]:
                raise AssertionError("Replayed native position does not match recorded training input")
            engine.send("nnueeval")
            probe = json.loads(engine.until(lambda line: line.startswith("info string NNUE_EVAL "))[-1].split("NNUE_EVAL ", 1)[1])
            if not probe["used"] or not probe["accumulator_matches"] or probe["white_cp"] != int(expected[i]):
                raise AssertionError(f"Integer/native parity failed at data row {index}: {probe}")
            report["tests"].append({"name": f"integer-parity/{index}", "passed": True,
                                    "game_id": row["game_id"], "ply": row["ply"], "white_cp": int(expected[i])})
            report["passed"] += 1
        # Search a few replayed heldout states; keep actual terminal/legality
        # handling in the native engine while the trained backend is active.
        for index in selected[:4]:
            row = rows[index]
            engine.position(games_by_id[row["game_id"]]["moves"][:row["ply"]])
            native = engine.features()
            if native["terminal"]["ended"]:
                continue
            bestmove, _ = engine.search(3)
            if bestmove not in native["legal_moves"]:
                raise AssertionError("Trained search returned an illegal move")
            engine.send("nnueeval")
            probe = json.loads(engine.until(lambda line: line.startswith("info string NNUE_EVAL "))[-1].split("NNUE_EVAL ", 1)[1])
            if not probe["used"] or not probe["accumulator_matches"]:
                raise AssertionError("Search make/unmake corrupted trained accumulator or disabled backend")
            report["tests"].append({"name": f"trained-search/{index}", "passed": True, "bestmove": bestmove})
            report["passed"] += 1
    except Exception as exc:
        report["failed"] += 1
        report["tests"].append({"name": "trained-runtime-validation", "passed": False, "error": str(exc)})
        raise
    finally:
        engine.close()
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        save_json(args.report, report)
    print(json.dumps({"variant": report["variant"], "passed": report["passed"], "failed": report["failed"],
                      "report": str(args.report)}))


if __name__ == "__main__":
    main()
