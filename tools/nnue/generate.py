#!/usr/bin/env python3
"""Generate reproducible classical-teacher data using native SVNNUE features.

Position rows are grouped by complete game. A max-plies stop is censored data,
never a draw. Teacher score labels remain usable for these truncated games.
This is a small CPU training experiment, not a strength or Elo measurement.

Copyright (C) 2026 Stockless Variants contributors
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
from queue import Empty, Queue
import random
import re
import subprocess
import sys
import threading
import time
from typing import Callable
from rich_features import feature_context


ROOT = Path(__file__).resolve().parents[2]
GENERATOR_VERSION = "svnnue-selfplay-v1"
VARIANTS = ("chess", "shogi", "janggi", "xiangqi", "makruk")
FEATURE_PREFIX = "info string NNUE_FEATURES "
LABEL_CP_LIMIT = 10000
CLAIM_POLICY = "claim_all_native_optional"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


class Engine:
    """One long-lived UCI process; every position replays its entire game history."""

    def __init__(self, executable: Path, timeout: float):
        self.timeout = timeout
        self.feature_version = 1
        self.process = subprocess.Popen(
            [str(executable), "noautoload"], cwd=executable.parent,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="strict", bufsize=1,
        )
        self.lines: Queue[str | None] = Queue()
        self.recent: deque[str] = deque(maxlen=30)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.send("uci")
            self.handshake = self.until(lambda line: line == "uciok")
        except BaseException:
            self.close()
            raise
        self.engine_id = next((line[8:] for line in self.handshake if line.startswith("id name ")), "unknown")

    def _read(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self.lines.put(line.rstrip("\r\n"))
        finally:
            self.lines.put(None)

    def send(self, text: str) -> None:
        if self.process.poll() is not None:
            raise RuntimeError(f"engine exited {self.process.returncode}: {list(self.recent)}")
        assert self.process.stdin is not None
        self.process.stdin.write(text + "\n")
        self.process.stdin.flush()

    def until(self, predicate: Callable[[str], bool]) -> list[str]:
        deadline = time.monotonic() + self.timeout
        result = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"engine response timed out; recent={list(self.recent)}")
            try:
                line = self.lines.get(timeout=remaining)
            except Empty as exc:
                raise RuntimeError(f"engine response timed out; recent={list(self.recent)}") from exc
            if line is None:
                raise RuntimeError(f"unexpected engine EOF; recent={list(self.recent)}")
            self.recent.append(line)
            if "info string ERROR" in line or line.startswith("ERROR"):
                raise RuntimeError(line)
            result.append(line)
            if predicate(line):
                return result

    def ready(self) -> None:
        self.send("isready")
        self.until(lambda line: line == "readyok")

    def option(self, name: str, value: object) -> None:
        self.send(f"setoption name {name} value {str(value).lower() if isinstance(value, bool) else value}")
        self.ready()

    def new_game(self) -> None:
        self.send("ucinewgame")
        self.ready()

    def position(self, moves: list[str]) -> None:
        self.send("position startpos" + (" moves " + " ".join(moves) if moves else ""))

    def features(self) -> dict:
        self.send("nnuefeaturesv2" if self.feature_version == 2 else "nnuefeatures")
        response = self.until(lambda line: line.startswith(FEATURE_PREFIX))[-1]
        state = json.loads(response[len(FEATURE_PREFIX):])
        if "error" in state:
            raise RuntimeError("native feature export: " + str(state["error"]))
        return state

    def search(self, depth: int) -> tuple[str, dict | None]:
        self.send(f"go depth {depth}")
        lines = self.until(lambda line: line.startswith("bestmove "))
        bestmove = lines[-1].split()[1]
        candidates = []
        for line in lines:
            if not line.startswith("info "):
                continue
            score = re.search(r"\bscore (cp|mate) (-?\d+)\b", line)
            actual_depth = re.search(r"\bdepth (\d+)\b", line)
            nodes = re.search(r"\bnodes (\d+)\b", line)
            multipv = re.search(r"\bmultipv (\d+)\b", line)
            if not (score and actual_depth and nodes) or (multipv and int(multipv[1]) != 1):
                continue
            bound = "lowerbound" if " lowerbound" in line else "upperbound" if " upperbound" in line else None
            candidates.append({
                "unit": score[1], "value": int(score[2]), "perspective": "side_to_move",
                "bound": bound, "depth": int(actual_depth[1]), "nodes": int(nodes[1]),
            })
        # Keep the last exact main-PV score at the highest completed depth.
        # A bound at the requested depth cannot be relabeled as an exact score.
        exact = [item for item in candidates if item["bound"] is None]
        label = max(enumerate(exact), key=lambda pair: (pair[1]["depth"], pair[0]))[1] if exact else None
        return bestmove, label

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait(timeout=5)
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()


def validate_state(state: dict, variant: str, expected: dict | None = None) -> dict:
    descriptor = state.get("descriptor", {})
    if state.get("schema") != 1 or descriptor.get("supported") is not True:
        raise ValueError("unsupported native export schema or descriptor")
    version = descriptor.get("format_version")
    if (descriptor.get("variant") != variant or type(version) is not int
            or version not in (1, 2) or descriptor.get("feature_version") != version):
        raise ValueError("native descriptor variant/feature version mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", str(descriptor.get("rules_sha256", ""))):
        raise ValueError("missing native rule digest")
    if expected is not None and descriptor != expected:
        raise ValueError("native descriptor changed during data generation")
    width, height, types = descriptor.get("width"), descriptor.get("height"), descriptor.get("type_ids")
    if not (type(width) is int and 1 <= width <= 12 and type(height) is int and 1 <= height <= 10
            and isinstance(types, list) and types
            and all(type(kind) is int and kind > 0 for kind in types)
            and types == sorted(set(types))):
        raise ValueError("invalid native board/type descriptor")
    area = width * height
    base = 2 * len(types) * (area + 1)
    dimension = base if version == 1 else base + 2 * len(types) * area + 4 + width + 6 * len(types)
    if (descriptor.get("input_dim") != dimension or descriptor.get("hidden_dim") != 64
            or descriptor.get("qa") != 256 or descriptor.get("qb") != 64):
        raise ValueError("native descriptor architecture mismatch")
    if type(state.get("side_to_move")) is not int or state["side_to_move"] not in (0, 1):
        raise ValueError("invalid native side to move")
    previous, total = -1, 0
    features = state.get("features")
    if not isinstance(features, list) or not features:
        raise ValueError("missing native features")
    for pair in features:
        if not (isinstance(pair, list) and len(pair) == 2 and all(type(v) is int for v in pair)):
            raise ValueError("invalid sparse feature pair")
        index, value = pair
        if not (previous < index < dimension and 0 < value <= 512):
            raise ValueError("out-of-range, duplicate, or unsorted native feature")
        if index < base and index % (width * height + 1) < width * height and value != 1:
            raise ValueError("board feature must have value one")
        previous, total = index, total + value
    if total > 512:
        raise ValueError("native feature L1 exceeds runtime safety bound")
    if not isinstance(state.get("fen"), str) or not state["fen"]:
        raise ValueError("missing native FEN audit text")
    moves = state.get("legal_moves")
    if (not isinstance(moves, list) or not all(isinstance(m, str) and m for m in moves)
            or len(set(moves)) != len(moves)):
        raise ValueError("invalid native legal move list")
    terminal = state.get("terminal", {})
    if type(terminal.get("ended")) is not bool or terminal.get("claim_policy") != CLAIM_POLICY:
        raise ValueError("invalid native terminal policy")
    if terminal["ended"]:
        if type(terminal.get("result_stm")) is not int or terminal["result_stm"] not in (-1, 0, 1):
            raise ValueError("native terminal result is missing")
        if terminal.get("reason") not in ("immediate_variant", "checkmate", "stalemate", "optional_rule"):
            raise ValueError("unknown native termination reason")
    elif terminal.get("result_stm") is not None or terminal.get("reason") is not None or not moves:
        raise ValueError("inconsistent ongoing native position")
    return descriptor


def score_label(score: dict, side: int) -> float:
    if score["unit"] == "mate":
        # Mate distance is retained in raw metadata; this is an explicit bounded
        # regression target, not a measured WDL probability or exact centipawn score.
        value = LABEL_CP_LIMIT if score["value"] > 0 else -LABEL_CP_LIMIT
    else:
        value = max(-LABEL_CP_LIMIT, min(LABEL_CP_LIMIT, score["value"]))
    return float(value if side == 0 else -value)


def build_metadata(executable: Path) -> tuple[dict | None, Path | None]:
    for candidate in (executable.parent.parent / "build-manifest.json", executable.parent / "build-manifest.json"):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8")), candidate
    return None, None


def declared_source_identity(build: dict | None, engine_id: str) -> tuple[str | None, bool | None]:
    """Match the executable's own identity to its adjacent build declaration.

    This establishes the declared base commit, not a reproducible-build proof.
    A dirty overlay is explicitly recorded and the actual binary is SHA hashed.
    """
    if not build or not re.fullmatch(r"[0-9a-f]{40}", str(build.get("commit", ""))):
        return None, None
    dirty = build.get("working_tree_dirty")
    if type(dirty) is not bool:
        return None, None
    identity = build["commit"][:12] + ("-dirty" if dirty else "")
    if f"({identity})" not in engine_id:
        return None, None
    return build["commit"], dirty


def generate(args: argparse.Namespace) -> dict:
    executable = args.engine.resolve()
    if not executable.is_file():
        raise ValueError(f"engine does not exist: {executable}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "positions.jsonl", "games.jsonl"):
        if (output / name).exists():
            raise ValueError(f"refusing to overwrite existing dataset file: {output / name}")
    binary_sha = sha256(executable)
    build, build_path = build_metadata(executable)
    started = time.monotonic()
    engine = Engine(executable, args.timeout)
    engine.feature_version = args.nnue_version
    generator_version = GENERATOR_VERSION if args.nnue_version == 1 else "svnnue-selfplay-v2"
    positions_path, games_path = output / "positions.jsonl", output / "games.jsonl"
    row_count = completed = truncated = skipped_labels = 0
    opening_plies = min(args.opening_plies, args.max_plies // 4)
    teacher_options = {
        "Threads": 1, "Hash": 16, "MultiPV": 1, "UCI_Chess960": False,
        "EvalBackend": "Classical", "Use NNUE": False, "StocklessHybrid": True,
        "UCI_Variant": args.variant,
    }
    descriptor = None
    try:
        for name, value in teacher_options.items():
            engine.option(name, value)
        with positions_path.open("x", encoding="utf-8") as positions_file, games_path.open("x", encoding="utf-8") as games_file:
            for game_index in range(args.games):
                seed_material = f"{generator_version}:{args.variant}:{args.seed}:{game_index}"
                game_seed = int.from_bytes(hashlib.sha256(seed_material.encode()).digest()[:8], "big")
                rng = random.Random(game_seed)
                engine.new_game()
                moves: list[str] = []
                rows: list[dict] = []
                state = None
                start_fen = None
                for ply in range(args.max_plies + 1):
                    engine.position(moves)
                    state = engine.features()
                    actual_descriptor = validate_state(state, args.variant, descriptor)
                    if descriptor is None:
                        descriptor = actual_descriptor
                    if start_fen is None:
                        start_fen = state["fen"]
                    if state["terminal"]["ended"] or ply == args.max_plies:
                        break
                    if ply < opening_plies:
                        moves.append(rng.choice(sorted(state["legal_moves"])))
                        continue
                    bestmove, score = engine.search(args.depth)
                    if bestmove not in state["legal_moves"]:
                        raise RuntimeError(f"teacher returned illegal move {bestmove!r} in {state['fen']}")
                    if score is None:
                        skipped_labels += 1
                    else:
                        fingerprint = hashlib.sha256(canonical({
                            "rules_sha256": descriptor["rules_sha256"],
                            "feature_version": descriptor["feature_version"],
                            "side_to_move": state["side_to_move"], "features": state["features"],
                        }).encode()).hexdigest()
                        rows.append({
                            "schema": 1, "ply": ply, "fen": state["fen"],
                            "variant": args.variant, "rules_sha256": descriptor["rules_sha256"],
                            "teacher_engine_sha256": binary_sha,
                            "side_to_move": state["side_to_move"], "features": state["features"],
                            "feature_sha256": fingerprint,
                            "label_cp_white": score_label(score, state["side_to_move"]),
                            "teacher": score,
                            **({"native_context_v2": feature_context(state, args.variant)}
                               if args.rich_context else {}),
                        })
                    moves.append(bestmove)
                assert state is not None and descriptor is not None and start_fen is not None
                game_id = hashlib.sha256(canonical({
                    "generator": generator_version, "variant": args.variant,
                    "rules_sha256": descriptor["rules_sha256"], "engine_sha256": binary_sha,
                    "seed": game_seed, "depth": args.depth, "max_plies": args.max_plies,
                    "opening_plies": opening_plies,
                }).encode()).hexdigest()
                ended = state["terminal"]["ended"]
                result_white = (state["terminal"]["result_stm"] * (1 if state["side_to_move"] == 0 else -1)) if ended else None
                reason = state["terminal"]["reason"] if ended else "max_plies"
                final = {"result_white": result_white, "terminal_reason": reason, "truncated": not ended}
                game = {
                    "schema": 1, "game_id": game_id, "game_index": game_index,
                    "variant": args.variant, "seed": game_seed, "start_fen": start_fen,
                    "rules_sha256": descriptor["rules_sha256"], "teacher_engine_sha256": binary_sha,
                    "moves": moves, "plies": len(moves), "opening_plies": min(opening_plies, len(moves)),
                    "final_fen": state["fen"], "final_side_to_move": state["side_to_move"],
                    "native_terminal": state["terminal"], "claim_policy": CLAIM_POLICY,
                    "rows": len(rows), **final,
                }
                games_file.write(canonical(game) + "\n")
                games_file.flush()
                for row in rows:
                    positions_file.write(canonical({**row, "game_id": game_id, **final}) + "\n")
                positions_file.flush()
                row_count += len(rows)
                completed += int(ended)
                truncated += int(not ended)
                print(f"{args.variant}: game {game_index + 1}/{args.games}, plies={len(moves)}, rows={len(rows)}, result={result_white}, truncated={not ended}", flush=True)
        if row_count == 0:
            raise RuntimeError("generation produced no exact-score training positions")
        source_commit, source_dirty = declared_source_identity(build, engine.engine_id)
        manifest = {
            "schema": 1, "generator_version": generator_version,
            "generator_source_commit": source_revision(), "variant": args.variant,
            "descriptor": descriptor, "engine_sha256": binary_sha,
            "source_commit": source_commit,
            "source_working_tree_dirty": source_dirty,
            "source_identity_basis": "executable id matched adjacent materialization manifest; actual binary separately SHA hashed"
                                     if source_commit else "unknown: no matching executable/build-manifest identity",
            "build_manifest_identity_matches": source_commit is not None,
            "upstream": build.get("upstream") if source_commit else None,
            "build_manifest_sha256": sha256(build_path) if build_path else None,
            "build_source_files_sha256": hashlib.sha256(canonical(build["source_files"]).encode()).hexdigest()
                                         if build and isinstance(build.get("source_files"), dict) else None,
            "rules_sha256": descriptor["rules_sha256"],
            "teacher": {
                "engine_id": engine.engine_id, "engine_sha256": binary_sha,
                "source_commit": source_commit, "source_working_tree_dirty": source_dirty,
                "evaluation": "classical", "options": teacher_options,
                "protocol": "UCI", "uci_option_defaults": [line for line in engine.handshake if line.startswith("option ")],
                "score_perspective": "side_to_move", "label_perspective": "white",
                "score_cp_definition": "native Value * 100 / 208; not variant pawn units",
                "label_cp_limit": LABEL_CP_LIMIT, "mate_target": "signed label_cp_limit; raw distance retained",
                "bound_policy": "last exact main-PV score at highest completed depth; otherwise skip position",
                "claim_policy": CLAIM_POLICY, "insufficient_material_adjudication": False,
            },
            "limits": {"games": args.games, "max_plies": args.max_plies, "depth": args.depth,
                       "opening_plies": opening_plies, "command_timeout_seconds": args.timeout},
            "seed": args.seed, "rows": row_count, "game_count": args.games,
            "completed_games": completed, "truncated_games": truncated, "skipped_labels": skipped_labels,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
                      for path in (positions_path, games_path)},
            "split_policy": "trainer must split by whole game and remove exact feature fingerprints crossing splits",
            "native_context_version": 2 if args.rich_context else None,
            "limitations": [
                "CPU pilot data for pipeline validation; no Elo or strength claim",
                "max-plies truncation has result null and is not a draw",
                "native optional results are automatically claimed, including non-draw variant results",
                "insufficient-material flags alone do not adjudicate a game",
                "features omit repetition history/counters; terminal decisions replay complete move history",
                "compact board features use native piece type and omit promoted-piece lineage",
                "same seed/settings support repeatability on the same engine/build; cross-platform identity is not promised",
            ],
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return manifest
    finally:
        engine.close()


def bounded_int(low: int, high: int) -> Callable[[str], int]:
    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError("expected an integer") from exc
        if not low <= number <= high:
            raise argparse.ArgumentTypeError(f"expected {low}..{high}")
        return number
    return parse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--games", type=bounded_int(1, 1024), default=32)
    parser.add_argument("--max-plies", type=bounded_int(4, 2048), default=80)
    parser.add_argument("--depth", type=bounded_int(1, 8), default=3)
    parser.add_argument("--seed", type=bounded_int(0, 2**63 - 1), default=20261026)
    parser.add_argument("--opening-plies", type=bounded_int(0, 32), default=8)
    parser.add_argument("--timeout", type=bounded_int(1, 600), default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nnue-version", type=bounded_int(1, 2), default=1,
                        help="1 stable SVNNUE-v1, 2 experimental native v2 inputs")
    parser.add_argument("--rich-context", action="store_true",
                        help="record native promoted lineage, rights and variant-region statistics for future SVNNUE-v2 work")
    args = parser.parse_args()
    if args.games * args.max_plies > 1_000_000:
        parser.error("one generation run is limited to one million plies")
    try:
        result = generate(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"generation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {result['rows']} positions from {result['game_count']} games to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
