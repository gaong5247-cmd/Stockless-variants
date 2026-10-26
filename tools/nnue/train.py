#!/usr/bin/env python3
"""Train a separate, small SVNNUE-v1 model on one variant's native self-play.

CPU pilot: classical-search distillation, not a strength-certified network.
Whole games are split before deduplication. Test labels never select checkpoints.
Copyright (C) 2026 Stockless Variants contributors
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np

from model import (HIDDEN, digest, float_predict, integer_predict, quantize,
                   read_network, validate_descriptor, write_network)

TRAINER_VERSION = "svnnue-numpy-adam-v1"
LABEL_LIMIT_CP = 2000.0
LOSS_SCALE_CP = 600.0
QUANTIZATION_LIMITS = {"mae_cp": 3.0, "p99_cp": 20.0, "max_cp": 40.0}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def feature_identity(row: dict) -> str:
    payload = json.dumps([row["side_to_move"], row["features"]], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_dataset(directory: Path) -> tuple[dict, list[dict], list[dict]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or type(manifest.get("schema")) is not int or manifest["schema"] != 1:
        raise ValueError("Unsupported dataset schema")
    validate_descriptor(manifest["descriptor"])
    if manifest.get("variant") != manifest["descriptor"]["variant"]:
        raise ValueError("Dataset variant and feature descriptor differ")
    d = manifest["descriptor"]
    if manifest.get("rules_sha256") != d["rules_sha256"]:
        raise ValueError("Dataset rules identity and feature descriptor differ")
    engine_hash = manifest.get("engine_sha256")
    if (not isinstance(engine_hash, str) or len(engine_hash) != 64
            or any(c not in "0123456789abcdef" for c in engine_hash)):
        raise ValueError("Dataset is missing its teacher binary identity")
    teacher_manifest = manifest.get("teacher", {})
    if (not isinstance(teacher_manifest, dict) or teacher_manifest.get("engine_sha256") != engine_hash
            or teacher_manifest.get("score_perspective") != "side_to_move"
            or teacher_manifest.get("label_perspective") != "white"
            or type(teacher_manifest.get("label_cp_limit")) is not int
            or teacher_manifest["label_cp_limit"] != 10000):
        raise ValueError("Unsupported teacher identity or score-label contract")
    claim_policy = "claim_all_native_optional"
    if teacher_manifest.get("claim_policy") != claim_policy:
        raise ValueError("Unsupported native result-claim policy")
    limits = manifest.get("limits", {})
    if not isinstance(limits, dict) or type(limits.get("max_plies")) is not int or limits["max_plies"] < 1:
        raise ValueError("Dataset is missing its game-length limit")
    for name in ("positions.jsonl", "games.jsonl"):
        expected = manifest["files"][name]
        if digest(directory / name) != expected["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
        if (directory / name).stat().st_size != expected["bytes"]:
            raise ValueError(f"Dataset byte count mismatch: {name}")
    games = [json.loads(line) for line in (directory / "games.jsonl").read_text().splitlines() if line]
    provenance = {"variant": manifest["variant"], "rules_sha256": d["rules_sha256"],
                  "teacher_engine_sha256": engine_hash}
    reasons = {"immediate_variant", "checkmate", "stalemate", "optional_rule"}
    game_by_id = {}
    completed = truncated = 0
    for game in games:
        if not isinstance(game, dict) or type(game.get("schema")) is not int or game["schema"] != 1:
            raise ValueError("Unsupported game record schema")
        if any(game.get(key) != value for key, value in provenance.items()):
            raise ValueError("Game variant/rules/teacher provenance differs from dataset")
        game_id = game.get("game_id")
        if not isinstance(game_id, str) or not game_id or game_id in game_by_id:
            raise ValueError("Duplicate or invalid game ID")
        plies, moves = game.get("plies"), game.get("moves")
        opening, expected_rows = game.get("opening_plies"), game.get("rows")
        if (type(plies) is not int or not 0 <= plies <= limits["max_plies"]
                or not isinstance(moves, list) or len(moves) != plies
                or not all(isinstance(move, str) and move and not any(c.isspace() for c in move) for move in moves)
                or type(opening) is not int or not 0 <= opening <= plies
                or type(expected_rows) is not int or not 0 <= expected_rows <= plies - opening):
            raise ValueError("Inconsistent game plies, moves, opening, or position count")
        is_truncated, result, reason = game.get("truncated"), game.get("result_white"), game.get("terminal_reason")
        final_side = game.get("final_side_to_move")
        native = game.get("native_terminal")
        if ("result_white" not in game or type(is_truncated) is not bool
                or type(final_side) is not int or final_side not in (0, 1)):
            raise ValueError("Invalid game truncation flag or final side")
        if result is not None and (type(result) is not int or result not in (-1, 0, 1)):
            raise ValueError("Invalid game result; expected null or -1/0/1")
        if (not isinstance(native, dict) or type(native.get("ended")) is not bool
                or native["ended"] == is_truncated or native.get("claim_policy") != claim_policy
                or game.get("claim_policy") != claim_policy):
            raise ValueError("Game/native terminal or claim policy differs")
        if is_truncated:
            if (result is not None or reason != "max_plies" or plies != limits["max_plies"]
                    or native.get("result_stm") is not None or native.get("reason") is not None):
                raise ValueError("Truncated game must retain null result and native ongoing state")
            truncated += 1
        else:
            native_result = native.get("result_stm")
            if (type(native_result) is not int or native_result not in (-1, 0, 1)
                    or result is None or reason not in reasons or native.get("reason") != reason
                    or result != native_result * (1 if final_side == 0 else -1)):
                raise ValueError("Completed game result differs from native signed terminal result")
            completed += 1
        game_by_id[game_id] = game
    game_ids = set(game_by_id)
    if type(manifest.get("game_count")) is not int or len(games) != manifest["game_count"]:
        raise ValueError("Duplicate/inconsistent game IDs")
    if (type(manifest.get("completed_games")) is not int or manifest["completed_games"] != completed
            or type(manifest.get("truncated_games")) is not int or manifest["truncated_games"] != truncated):
        raise ValueError("Dataset completed/truncated game counts differ from records")
    rows = []
    game_plies = {game_id: set() for game_id in game_ids}
    area = d["width"] * d["height"]
    with (directory / "positions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict) or type(row.get("schema")) is not int or row["schema"] != 1:
                raise ValueError("Unsupported position record schema")
            if any(row.get(key) != value for key, value in provenance.items()):
                raise ValueError("Position variant/rules/teacher provenance differs from dataset")
            game_id = row.get("game_id")
            if (not isinstance(game_id, str) or game_id not in game_ids
                    or type(row.get("side_to_move")) is not int or row["side_to_move"] not in (0, 1)):
                raise ValueError("Position has unknown game or invalid side")
            game, ply = game_by_id[game_id], row.get("ply")
            if (type(ply) is not int or not game["opening_plies"] <= ply < game["plies"]
                    or ply in game_plies[game_id]):
                raise ValueError("Duplicate or out-of-range game position ply")
            game_plies[game_id].add(ply)
            label = row.get("label_cp_white")
            if type(label) not in (float, int) or not -10000 <= label <= 10000 or not math.isfinite(label):
                raise ValueError("Invalid teacher label")
            score = row.get("teacher")
            if (not isinstance(score, dict) or score.get("unit") not in ("cp", "mate")
                    or type(score.get("value")) is not int or score.get("perspective") != "side_to_move"
                    or type(score.get("depth")) is not int or score["depth"] < 1
                    or type(score.get("nodes")) is not int or score["nodes"] < 1 or "bound" not in score):
                raise ValueError("Missing or invalid raw teacher score metadata")
            if score["bound"] is not None:
                raise ValueError("A search bound cannot be used as an exact label")
            expected_label = (10000 if score["value"] > 0 else -10000) if score["unit"] == "mate" \
                             else max(-10000, min(10000, score["value"]))
            expected_label *= 1 if row["side_to_move"] == 0 else -1
            if label != expected_label:
                raise ValueError("White label does not match the raw side-to-move teacher score")
            row_result = row.get("result_white")
            if row_result is not None and (type(row_result) is not int or row_result not in (-1, 0, 1)):
                raise ValueError("Invalid position result; expected null or -1/0/1")
            if ("result_white" not in row or type(row.get("truncated")) is not bool
                    or row["truncated"] != game["truncated"] or row_result != game["result_white"]
                    or row.get("terminal_reason") != game["terminal_reason"]):
                raise ValueError("Position/game result, truncation, or terminal reason differs")
            previous, norm = -1, 0
            features = row.get("features")
            if not isinstance(features, list):
                raise ValueError("Position features must be sparse index/value pairs")
            for pair in features:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise ValueError("Position features must be sparse index/value pairs")
                index, value = pair
                if type(index) is not int or type(value) is not int or not previous < index < d["input_dim"]:
                    raise ValueError("Features must have sorted, unique, in-range indices")
                base_dim = 2 * len(d["type_ids"]) * (area + 1)
                if value < 1 or (index < base_dim and index % (area + 1) < area and value != 1):
                    raise ValueError("Invalid board indicator or pocket count")
                norm += abs(value)
                previous = index
            if not row["features"] or norm > 512:
                raise ValueError("Feature activation exceeds runtime bound")
            row["input_sha256"] = feature_identity(row)
            rows.append(row)
            if len(rows) > 100000:
                raise ValueError("CPU pilot is limited to 100,000 positions per model")
    if type(manifest.get("rows")) is not int or len(rows) != manifest["rows"]:
        raise ValueError("Dataset row count mismatch")
    if any(len(game_plies[game_id]) != game_by_id[game_id]["rows"] for game_id in game_ids):
        raise ValueError("Per-game position count differs from records")
    return manifest, rows, games


def partition(rows: list[dict], games: list[dict], seed: int) -> tuple[dict, dict]:
    # Group by complete game first; adjacent states never cross partitions.
    available = {row["game_id"] for row in rows}
    game_ids = sorted(game["game_id"] for game in games if game["game_id"] in available)
    if len(game_ids) < 6:
        raise ValueError("At least six nonempty games are needed for train/validation/test")
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(game_ids))
    holdout = max(1, len(shuffled) // 6)
    groups = {"train": shuffled[2 * holdout:], "validation": shuffled[:holdout],
              "test": shuffled[holdout:2 * holdout]}
    assigned = {game: split for split, ids in groups.items() for game in ids}
    parts = {name: [] for name in groups}
    removed = {name: 0 for name in groups}
    seen = set()
    # Exact identical model inputs are retained in only one partition. Training
    # gets priority; holdouts cannot accidentally test an already-seen opening.
    for name in ("train", "validation", "test"):
        for index, row in enumerate(rows):
            if assigned[row["game_id"]] != name:
                continue
            key = row["input_sha256"]
            if key in seen:
                removed[name] += 1
            else:
                seen.add(key)
                parts[name].append(index)
        if len(parts[name]) < 8:
            raise ValueError(f"Too few unique {name} positions after game split/deduplication")
    report = {"method": "whole-game 2/3 train, 1/6 validation, 1/6 test; minimum one heldout game",
              "seed": seed, "game_ids": {k: [str(v) for v in vals] for k, vals in groups.items()},
              "position_indices": parts, "retained_positions": {k: len(v) for k, v in parts.items()},
              "removed_duplicate_inputs": removed, "cross_split_input_overlap": 0,
              "test_labels_used_for_checkpoint_selection": False}
    return {key: np.asarray(values, dtype=np.int64) for key, values in parts.items()}, report


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    residual = prediction.astype(np.float64) - target.astype(np.float64)
    if not np.isfinite(residual).all():
        raise ValueError("Nonfinite predictions/loss")
    absolute = np.abs(residual)
    return {"positions": len(target), "mse_cp2": float(np.mean(residual ** 2)),
            "rmse_cp": float(np.sqrt(np.mean(residual ** 2))), "mae_cp": float(np.mean(absolute))}


def weight_digest(weights: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for key in ("w1", "b1", "w2", "b2"):
        h.update(key.encode())
        h.update(np.asarray(weights[key], dtype="<f4").tobytes())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 500 or not 8 <= args.batch_size <= 1024 or not 0 <= args.seed < 2 ** 32:
        parser.error("epochs 1..500, batch size 8..1024 and uint32 seed required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty; experiments must not silently overwrite a model")
    started = time.monotonic()
    dataset, rows, games = load_dataset(args.data)
    d = dataset["descriptor"]
    parts, split = partition(rows, games, args.seed)
    if len(rows) * d["input_dim"] * 4 > 512 * 1024 * 1024:
        raise ValueError("Dense CPU pilot exceeds its 512 MiB input budget; generate fewer positions")
    x = np.zeros((len(rows), d["input_dim"]), dtype=np.float32)
    side = np.asarray([row["side_to_move"] for row in rows], dtype=np.int64)
    raw_targets = np.asarray([row["label_cp_white"] for row in rows], dtype=np.float32)
    y = np.clip(raw_targets, -LABEL_LIMIT_CP, LABEL_LIMIT_CP)
    for i, row in enumerate(rows):
        for index, value in row["features"]:
            x[i, index] = value
    rng = np.random.default_rng(args.seed)
    weights = {"w1": rng.normal(0.0, 0.025, (d["input_dim"], HIDDEN)).astype(np.float32),
               "b1": np.full(HIDDEN, 0.25, dtype=np.float32),
               "w2": rng.normal(0.0, 5.0, (2, HIDDEN)).astype(np.float32),
               "b2": np.zeros(2, dtype=np.float32)}
    # Label centering is fitted solely on training games, separately by head.
    for color in (0, 1):
        selected = parts["train"][side[parts["train"]] == color]
        if len(selected):
            weights["b2"][color] = np.mean(y[selected])
    initial_digest = weight_digest(weights)
    initial = {name: metrics(float_predict(weights, x[idx], side[idx]), y[idx])
               for name, idx in parts.items() if name != "test"}
    moment = {k: np.zeros_like(v) for k, v in weights.items()}
    variance = {k: np.zeros_like(v) for k, v in weights.items()}
    rates = {"w1": 0.008, "b1": 0.008, "w2": 1.5, "b2": 1.5}
    history = []
    best, best_loss, best_epoch = None, float("inf"), None
    steps = 0
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / "split.json", split)

    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(parts["train"])
        for offset in range(0, len(order), args.batch_size):
            selected = order[offset:offset + args.batch_size]
            batch, heads, target = x[selected], side[selected], y[selected]
            pre = batch @ weights["w1"] + weights["b1"]
            activation = np.clip(pre, 0, 1)
            prediction = np.sum(activation * weights["w2"][heads], axis=1) + weights["b2"][heads]
            gradient = (prediction - target) * (2.0 / (len(selected) * LOSS_SCALE_CP ** 2))
            hidden_grad = gradient[:, None] * weights["w2"][heads] * ((pre > 0) & (pre < 1))
            gradients = {"w1": batch.T @ hidden_grad + 1e-6 * weights["w1"],
                         "b1": np.sum(hidden_grad, axis=0),
                         "w2": np.zeros_like(weights["w2"]), "b2": np.zeros_like(weights["b2"])}
            for color in (0, 1):
                mask = heads == color
                gradients["w2"][color] = np.sum(activation[mask] * gradient[mask, None], axis=0)
                gradients["b2"][color] = np.sum(gradient[mask])
            steps += 1
            decay = 0.25 + 0.75 * (1.0 - (epoch - 1) / args.epochs)
            for key in weights:
                g = gradients[key]
                if not np.isfinite(g).all():
                    raise ValueError(f"Nonfinite gradient in {key}")
                moment[key] = 0.9 * moment[key] + 0.1 * g
                variance[key] = 0.999 * variance[key] + 0.001 * g * g
                corrected = moment[key] / (1.0 - 0.9 ** steps)
                denominator = np.sqrt(variance[key] / (1.0 - 0.999 ** steps)) + 1e-8
                weights[key] -= rates[key] * decay * corrected / denominator
            # Explicit optimizer projection keeps exported integer parameters
            # representable; counts and exact scales are part of the format.
            np.clip(weights["w1"], -8, 8, out=weights["w1"])
            np.clip(weights["b1"], -8, 8, out=weights["b1"])
            np.clip(weights["w2"], -500, 500, out=weights["w2"])
            np.clip(weights["b2"], -10000, 10000, out=weights["b2"])
        current = {name: metrics(float_predict(weights, x[idx], side[idx]), y[idx])
                   for name, idx in parts.items() if name != "test"}
        history.append({"epoch": epoch, "steps": steps, **current})
        if current["validation"]["mse_cp2"] < best_loss:
            best_loss = current["validation"]["mse_cp2"]
            best_epoch = epoch
            best = {k: v.copy() for k, v in weights.items()}
        print(json.dumps({"variant": d["variant"], "epoch": epoch,
                          "train_mae_cp": round(current["train"]["mae_cp"], 3),
                          "validation_mae_cp": round(current["validation"]["mae_cp"], 3)}), flush=True)

    assert best is not None and best_epoch is not None
    final_digest = weight_digest(best)
    if final_digest == initial_digest or not steps:
        raise ValueError("No actual parameter update occurred")
    q = quantize(best, d)
    network = args.output / (d["variant"] + ".svnnue")
    write_network(network, d, q)
    reloaded = read_network(network, d)
    if any(not np.array_equal(q[k], reloaded[k]) for k in q):
        raise ValueError("Network serialization round trip changed weights")
    np.savez_compressed(args.output / "checkpoint.npz", **best,
                        descriptor=np.asarray(json.dumps(d, sort_keys=True)),
                        selected_epoch=np.asarray(best_epoch), trainer_version=np.asarray(TRAINER_VERSION))
    final = {name: metrics(float_predict(best, x[idx], side[idx]), y[idx]) for name, idx in parts.items()}
    quantized = {}
    quant_errors = {}
    for name in ("validation", "test"):
        idx = parts[name]
        integer = integer_predict(reloaded, x[idx], side[idx])
        floating = float_predict(best, x[idx], side[idx])
        error = np.abs(integer.astype(np.float64) - floating.astype(np.float64))
        quantized[name] = metrics(integer, y[idx])
        quant_errors[name] = {"mae_cp": float(error.mean()), "p99_cp": float(np.quantile(error, 0.99)),
                              "max_cp": float(error.max(initial=0))}
    passed = all(value <= QUANTIZATION_LIMITS[key]
                 for values in quant_errors.values() for key, value in values.items())
    mean_train = float(np.mean(y[parts["train"]]))
    report = {"schema": 1, "trainer_version": TRAINER_VERSION, "descriptor": d,
              "dataset_manifest_sha256": digest(args.data / "manifest.json"),
              "data_files": dataset["files"], "teacher": dataset["teacher"],
              "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
              "optimizer": "Adam; beta1=.9,beta2=.999; train-only gradients; no GPU",
              "base_learning_rates": rates, "loss": "MSE on clipped white centipawns / 600^2",
              "label_clip_cp": LABEL_LIMIT_CP, "clipped_label_rows": int(np.count_nonzero(y != raw_targets)),
              "outcomes_used_in_loss": False, "gradient_steps": steps, "selected_epoch": best_epoch,
              "checkpoint_selection": "lowest validation MSE among trained epochs; test unused",
              "initial_weights_sha256": initial_digest, "trained_weights_sha256": final_digest,
              "initial": initial, "final_float": final, "final_quantized": quantized,
              "test_train_mean_baseline": metrics(np.full(len(parts["test"]), mean_train), y[parts["test"]]),
              "validation_improved_from_initial": final["validation"]["mse_cp2"] < initial["validation"]["mse_cp2"],
              "quantization_error": quant_errors, "quantization_limits": QUANTIZATION_LIMITS,
              "quantization_passed": passed, "split": {k: v for k, v in split.items() if k != "position_indices"},
              "history": history, "elapsed_seconds": round(time.monotonic() - started, 3),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "platform": platform.platform(), "threads": {k: os.environ.get(k) for k in
                          ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}},
              "status": "experimental-classical-teacher-distillation", "elo": None,
              "strength_validated": False, "training_completed": True}
    save_json(args.output / "metrics.json", report)
    model_manifest = {"schema": 1, "format": "SVNNUE-v1", "descriptor": d,
                      "network": network.name, "network_sha256": digest(network),
                      "network_bytes": network.stat().st_size, "license": "AGPL-3.0-only",
                      "source_repository": "https://github.com/gaong5247-cmd/Stockless-variants",
                      "teacher": dataset["teacher"], "teacher_engine_sha256": dataset["engine_sha256"],
                      "dataset_manifest_sha256": digest(args.data / "manifest.json"),
                      "trainer_version": TRAINER_VERSION, "training_seed": args.seed,
                      "selected_epoch": best_epoch, "metrics_sha256": digest(args.output / "metrics.json"),
                      "checkpoint_sha256": digest(args.output / "checkpoint.npz"),
                      "split_sha256": digest(args.output / "split.json"),
                      "status": "experimental", "default_enabled": False, "strength_validated": False,
                      "quantization_passed": passed}
    save_json(args.output / "network-manifest.json", model_manifest)
    print(json.dumps({"network": str(network), "gradient_steps": steps, "selected_epoch": best_epoch,
                      "test_mae_cp": final["test"]["mae_cp"], "quantization_passed": passed}), flush=True)
    if not passed:
        raise SystemExit("Quantization accuracy gate failed; experiment artifacts retained for inspection")


if __name__ == "__main__":
    main()
