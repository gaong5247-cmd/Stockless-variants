#!/usr/bin/env python3
"""Independent SVNNUE-v1 file, feature and integer-inference acceptance tests.

This runner writes deterministic SYNTHETIC fixture networks, compares engine
inference with Python integer arithmetic, tests transactional loader rejection,
and optionally runs the separately compiled C++ make/undo accumulator harness.
It requires only Python's standard library. It does not measure playing strength
or claim that synthetic fixture weights are trained networks.

Example:
  python3 tests/nnue_validation_test.py --engine build/engine/src/stockless-variants \
      --cpp-test build/tests/nnue_test --fixture-dir build/nnue-fixtures \
      --report build/nnue-tests.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import re
import struct
import subprocess
import time
from typing import Callable

from run_engine_tests import Engine, ROOT, sha256


GAMES = ("chess", "shogi", "janggi", "xiangqi", "makruk")
HIDDEN, QA, QB, HEADER_BYTES = 64, 256, 64, 160
MAGIC = b"SVNNUE1\0"
HEADER = struct.Struct("<8s10I32s32s32s16s")
assert HEADER.size == HEADER_BYTES

# Native PieceType IDs from the pinned rules interface. In particular the Shogi
# pawn/knight, Janggi cannon/elephant and Makruk khon are not orthodox chess IDs.
EXPECTED_TYPES = {
    "chess": [1, 2, 3, 4, 5, 63],
    "shogi": [3, 4, 9, 11, 19, 20, 21, 22, 23, 63],
    "janggi": [4, 28, 29, 30, 32, 34, 63],
    "xiangqi": [4, 6, 27, 29, 30, 31, 63],
    "makruk": [1, 2, 4, 6, 9, 63],
}
GEOMETRY = {"chess": (8, 8), "shogi": (9, 9), "janggi": (9, 10),
            "xiangqi": (9, 10), "makruk": (8, 8)}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def query(engine: Engine, command: str, marker: str) -> dict:
    prefix = "info string " + marker + " "
    engine.send(command)
    lines = engine.until(lambda line: line.startswith(prefix))
    result = json.loads(lines[-1][len(prefix):])
    require(isinstance(result, dict), command + " did not return a JSON object")
    return result


def features(engine: Engine) -> dict:
    result = query(engine, "nnuefeatures", "NNUE_FEATURES")
    require("error" not in result, "feature export refused position: " + str(result))
    return result


def probe(engine: Engine) -> dict:
    return query(engine, "nnueeval", "NNUE_EVAL")


def status(engine: Engine) -> dict:
    return query(engine, "nnuestatus", "NNUE_STATUS")


def set_option(engine: Engine, name: str, value: str | Path | bool) -> list[str]:
    if isinstance(value, bool):
        value = str(value).lower()
    engine.send(f"setoption name {name} value {value}")
    return engine.ready()


def successful_option(engine: Engine, name: str, value: str | Path | bool) -> None:
    output = set_option(engine, name, value)
    require(not any("ERROR" in line for line in output),
            f"{name} rejected {value}: " + "\n".join(output))


def activate(engine: Engine, filename: Path) -> dict:
    successful_option(engine, "EvalFile", filename.resolve())
    successful_option(engine, "EvalBackend", "StocklessNNUE")
    result = status(engine)
    require(result.get("loaded") is True and result.get("active") is True
            and result.get("compatible") is True and result.get("backend") == "StocklessNNUE",
            "successful load did not activate the actual neural backend: " + str(result))
    return result


def validate_descriptor(desc: dict, variant: str) -> None:
    require(desc.get("supported") is True and desc.get("variant") == variant,
            f"wrong descriptor for {variant}: {desc}")
    for key, expected in (("format_version", 1), ("feature_version", 1),
                          ("hidden_dim", HIDDEN), ("qa", QA), ("qb", QB)):
        require(desc.get(key) == expected, f"{variant}: wrong {key}")
    require((desc.get("width"), desc.get("height")) == GEOMETRY[variant], "wrong board geometry")
    require(desc.get("type_ids") == EXPECTED_TYPES[variant],
            f"{variant}: native piece layout changed: {desc.get('type_ids')}")
    require(re.fullmatch("[0-9a-f]{64}", desc.get("rules_sha256", "")) is not None,
            "missing deterministic rules fingerprint")
    area = desc["width"] * desc["height"]
    require(desc.get("input_dim") == 2 * len(desc["type_ids"]) * (area + 1), "wrong feature dimension")


def independent_features(state: dict) -> list[list[int]]:
    """Derive dense-board indices from semantic fields, never engine indices.

    Directed fixtures below independently check those semantic fields too, so
    two equally wrong engine exports cannot make the entire suite pass.
    """
    desc = state["descriptor"]
    width, height = desc["width"], desc["height"]
    area, kinds = width * height, desc["type_ids"]
    require(kinds == sorted(set(kinds)), "unsorted or duplicate native type IDs")
    mapped: Counter[int] = Counter()
    occupied = set()
    for piece in state["pieces"]:
        file, rank, color, kind = (piece[key] for key in ("file", "rank", "color", "type"))
        require(0 <= file < width and 0 <= rank < height, "piece outside game board")
        require(color in (0, 1) and kind in kinds, "invalid piece color or native ID")
        require((file, rank) not in occupied, "two pieces occupy one square")
        occupied.add((file, rank))
        index = ((color * len(kinds) + kinds.index(kind)) * (area + 1) + rank * width + file)
        mapped[index] += 1
    seen_pockets = set()
    for pocket in state["pockets"]:
        color, kind, count = (pocket[key] for key in ("color", "type", "count"))
        require(color in (0, 1) and kind in kinds and isinstance(count, int) and count > 0,
                "invalid pocket feature")
        require((color, kind) not in seen_pockets, "duplicate pocket entry")
        seen_pockets.add((color, kind))
        mapped[(color * len(kinds) + kinds.index(kind)) * (area + 1) + area] += count
    expected = [[index, value] for index, value in sorted(mapped.items()) if value]
    actual = state["features"]
    require(actual == expected, f"dense per-game feature mapping differs: {actual} != {expected}")
    require(all(0 <= index < desc["input_dim"] for index, _ in actual), "feature index out of range")
    require(sum(abs(value) for _, value in actual) <= 512, "feature mass exceeds arithmetic contract")
    require(state.get("side_to_move") in (0, 1), "invalid side to move")
    return expected


def round_away(numerator: int, divisor: int) -> int:
    return ((numerator + divisor // 2) // divisor if numerator >= 0
            else -((-numerator + divisor // 2) // divisor))


@dataclass
class Model:
    descriptor: dict
    w1: list[int]
    b1: list[int]
    w2: list[int]
    b2: list[int]

    def payload(self) -> bytes:
        return (struct.pack(f"<{len(self.w1)}h", *self.w1)
                + struct.pack("<64i", *self.b1)
                + struct.pack("<128h", *self.w2)
                + struct.pack("<2i", *self.b2))

    def binary(self) -> bytes:
        desc, payload = self.descriptor, self.payload()
        name = desc["variant"].encode("ascii")
        require(len(name) < 32, "fixture variant name too long")
        return HEADER.pack(MAGIC, 1, 1, HIDDEN, desc["input_dim"], desc["width"],
                           desc["height"], len(desc["type_ids"]), QA, QB, len(payload),
                           name.ljust(32, b"\0"), bytes.fromhex(desc["rules_sha256"]),
                           hashlib.sha256(payload).digest(), bytes(16)) + payload

    def predict(self, state: dict) -> dict:
        sparse = independent_features(state)
        accumulator = self.b1.copy()
        for index, count in sparse:
            start = index * HIDDEN
            for hidden in range(HIDDEN):
                accumulator[hidden] += count * self.w1[start + hidden]
        clipped = [max(0, min(QA, value)) for value in accumulator]
        side = state["side_to_move"]
        numerator = self.b2[side] + sum(
            clipped[hidden] * self.w2[side * HIDDEN + hidden] for hidden in range(HIDDEN))
        cp = max(-10000, min(10000, round_away(numerator, QA * QB)))
        # Binary-grid float inference independently checks the scaling convention.
        # This is exact-grid parity, not a trained-model quantization quality gate.
        float_cp = self.b2[side] / (QA * QB)
        float_cp += sum(max(0.0, min(1.0, value / QA))
                        * (self.w2[side * HIDDEN + hidden] / QB)
                        for hidden, value in enumerate(accumulator))
        require(float_cp == numerator / (QA * QB), "integer/float scale contract differs")
        return {"accumulator": accumulator, "numerator": numerator, "white_cp": cp,
                "stm_cp": cp if side == 0 else -cp}


def synthetic_model(desc: dict, seed: int = 1, mode: str = "pattern") -> Model:
    dimension = desc["input_dim"]
    if mode == "pattern":
        weights = [((index * 17 + hidden * 7 + (index >> 3) * (hidden + 3) + seed * 29) % 31) - 15
                   for index in range(dimension) for hidden in range(HIDDEN)]
        biases = [((hidden * 37 + seed * 53) % 420) - 80 for hidden in range(HIDDEN)]
        output = [(((hidden * 19 + head * 31 + seed * 11) % 129) - 64) * 23
                  for head in range(2) for hidden in range(HIDDEN)]
        output_bias = [seed * 3271 + 8192, -seed * 9187 - 8192]
    elif mode == "extreme":
        weights = [32767 if hidden % 2 == 0 else -32768
                   for _ in range(dimension) for hidden in range(HIDDEN)]
        biases = [2**20 if hidden % 2 == 0 else -(2**20) for hidden in range(HIDDEN)]
        output = [32767] * HIDDEN + [-32768] * HIDDEN
        output_bias = [2**31 - 1, -(2**31)]
    else:
        weights = [0] * (dimension * HIDDEN)
        biases, output = [0] * HIDDEN, [0] * (HIDDEN * 2)
        output_bias = {"rounding": [8192, -8192], "below-half": [8191, -8191],
                       "saturation": [2**31 - 1, -(2**31)]}[mode]
    return Model(desc, weights, biases, output, output_bias)


def check_prediction(engine: Engine, model: Model, state: dict | None = None) -> dict:
    state = state or features(engine)
    expected = model.predict(state)
    actual = probe(engine)
    require(actual.get("used") is True and actual.get("compatible") is True
            and actual.get("active") is True, "inference silently fell back: " + str(actual))
    require(actual.get("backend") in ("StocklessNNUE", "Auto")
            and actual.get("model_variant") == state["descriptor"]["variant"],
            "wrong backend or game network actually evaluated")
    require(actual.get("accumulator_matches") is True, "incremental/full-refresh disagreement")
    for key in ("accumulator", "numerator", "white_cp", "stm_cp"):
        require(actual.get(key) == expected[key],
                f"Python/C++ {key} differs: {actual.get(key)} != {expected[key]}")
    require(actual.get("refreshed_accumulator") == expected["accumulator"],
            "independent Python/full-refresh vector differs")
    # Pinned FSF PawnValueEg=208; Python // would round negatives incorrectly,
    # so reproduce C++ truncation toward zero explicitly.
    scaled = abs(expected["stm_cp"]) * 208 // 100
    require(actual.get("engine_value") == (scaled if expected["stm_cp"] >= 0 else -scaled),
            "centipawn to engine Value conversion differs")
    require(actual.get("generation", 0) > 0, "unversioned accumulator")
    return actual


def semantic_pieces(state: dict) -> set[tuple[int, int, int, int]]:
    return {(p["file"], p["rank"], p["color"], p["type"]) for p in state["pieces"]}


def mutated_files(good: bytes, dimension: int) -> dict[str, bytes]:
    result = {
        "empty-file": b"", "short-header": good[:159], "header-only": good[:160],
        "truncated-input": good[:160 + dimension * HIDDEN * 2 - 1],
        "truncated-last-byte": good[:-1], "trailing-byte": good + b"\0",
        "oversized-file": good + bytes(4 * 1024 * 1024),
    }

    def change(name: str, offset: int, value: bytes) -> None:
        changed = bytearray(good)
        changed[offset:offset + len(value)] = value
        result[name] = bytes(changed)

    change("native-fsf-magic", 0, b"HalfKAv1")
    for name, offset, value in (
        ("format-version", 8, 2), ("feature-layout-version", 12, 2),
        ("hidden-width", 16, 32), ("input-dimension", 20, dimension + 1),
        ("huge-input-dimension", 20, 0xFFFFFFFF), ("board-width", 24, 9),
        ("board-height", 28, 9), ("native-type-count", 32, 7),
        ("activation-quantization", 36, 128), ("output-quantization", 40, 128),
        ("declared-payload-size", 44, len(good) - HEADER_BYTES + 1),
    ):
        change(name, offset, struct.pack("<I", value))
    change("same-size-wrong-variant", 48, b"makruk".ljust(32, b"\0"))
    change("variant-name-padding", 79, b"!")
    change("wrong-rules-fingerprint", 80, bytes([good[80] ^ 0x80]))
    change("reserved-header-byte", 144, b"\x01")
    change("payload-checksum", 112, bytes([good[112] ^ 0x80]))
    change("corrupt-input-weight", 160, bytes([good[160] ^ 1]))
    change("corrupt-output-weight", len(good) - 10, bytes([good[-10] ^ 1]))
    for name, bias in (("resealed-excess-positive-input-bias", 2**20 + 1),
                       ("resealed-excess-negative-input-bias", -(2**20) - 1)):
        changed = bytearray(good)
        offset = 160 + dimension * HIDDEN * 2
        changed[offset:offset + 4] = struct.pack("<i", bias)
        changed[112:144] = hashlib.sha256(changed[160:]).digest()
        result[name] = bytes(changed)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--cpp-test", type=Path, help="compiled tests/nnue_test.cpp harness")
    parser.add_argument("--fixture-dir", type=Path, default=ROOT / "build/nnue-fixtures")
    parser.add_argument("--report", type=Path, default=ROOT / "build/nnue-tests.json")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--fixtures-only", action="store_true", help="generate fixtures; not an acceptance run")
    args = parser.parse_args()
    args.fixture_dir = args.fixture_dir.resolve()
    args.fixture_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {
        "schema": 1, "suite": "SVNNUE-v1 independent runtime validation",
        "engine": str(args.engine.resolve()), "engine_sha256": sha256(args.engine),
        "platform": platform.platform(), "python": platform.python_version(),
        "network_origin": "deterministic synthetic fixtures; not trained networks",
        "cpp_harness_requested": args.cpp_test is not None,
        "fixtures_only": args.fixtures_only, "descriptors": {}, "fixture_sha256": {}, "tests": [],
        "limitations": ["no playing-strength claim", "bounded legal-position coverage",
                        "float parity here uses exactly quantized fixture weights; trained-model heldout quantization gates run separately",
                        "header binds native type order by source fingerprint and type count; weights alone cannot reveal false provenance"],
    }

    def run(name: str, body: Callable[[], dict | None]) -> bool:
        start = time.monotonic()
        try:
            evidence = body() or {}
            result = {"name": name, "passed": True, "evidence": evidence}
        except Exception as exc:
            result = {"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        result["elapsed_seconds"] = round(time.monotonic() - start, 3)
        report["tests"].append(result)
        print(("PASS " if result["passed"] else "FAIL ") + name, flush=True)
        if not result["passed"]:
            print(result["error"], flush=True)
        return result["passed"]

    models: dict[str, Model] = {}

    def write_model(name: str, model: Model) -> Path:
        path = args.fixture_dir / (name + ".svnnue")
        path.write_bytes(model.binary())
        models[name] = model
        report["fixture_sha256"][name] = sha256(path)
        return path

    def model_path(variant: str, suffix: str = "a") -> Path:
        return args.fixture_dir / f"{variant}-{suffix}.svnnue"

    def generate() -> dict:
        with Engine(args.engine, timeout=args.timeout) as e:
            for variant in GAMES:
                e.variant(variant)
                e.position()
                state = features(e)
                desc = state["descriptor"]
                validate_descriptor(desc, variant)
                independent_features(state)
                report["descriptors"][variant] = desc
                write_model(variant + "-a", synthetic_model(desc, 1))
                write_model(variant + "-b", synthetic_model(desc, 7))
            chess = report["descriptors"]["chess"]
            for mode in ("rounding", "below-half", "saturation", "extreme"):
                write_model("chess-" + mode, synthetic_model(chess, mode=mode))
        return {"variants": list(GAMES), "files": len(models), "directory": str(args.fixture_dir)}

    generated = run("fixtures/native-descriptors-and-independent-packing", generate)
    if not generated or args.fixtures_only:
        return finish(args.report, report, started)

    def default_and_missing_model() -> dict:
        with Engine(args.engine, timeout=args.timeout) as e:
            e.position()
            before = status(e)
            require(before.get("backend") == "Classical" and before.get("loaded") is False,
                    "default engine must remain Classical without a model")
            require(probe(e).get("used") is False, "unloaded backend evaluated NNUE")
            output = set_option(e, "EvalBackend", "StocklessNNUE")
            require(any("ERROR" in line for line in output), "missing-network activation silently accepted")
            require(status(e).get("backend") == "Classical", "failed activation changed default backend")
            successful_option(e, "EvalFile", model_path("chess"))
            require(status(e).get("loaded") is True and probe(e).get("used") is False,
                    "loading alone unexpectedly enabled NNUE")
            successful_option(e, "EvalBackend", "StocklessNNUE")
            check_prediction(e, models["chess-a"])
        return {"default": "Classical", "loading_requires_activation": True}

    run("activation/default-load-and-explicit-opt-in", default_and_missing_model)

    def golden_walk(variant: str) -> dict:
        with Engine(args.engine, timeout=args.timeout) as e:
            e.variant(variant)
            e.position()
            active = activate(e, model_path(variant))
            moves, seen, numerators = [], set(), set()
            seed = 0x53564E4E
            for ply in range(33):
                e.position(moves=moves)
                state = features(e)
                validate_descriptor(state["descriptor"], variant)
                actual = check_prediction(e, models[variant + "-a"], state)
                numerators.add(actual["numerator"])
                seen.update(index for index, _ in state["features"])
                if state["terminal"]["ended"] or not state["legal_moves"]:
                    break
                seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
                legal = sorted(state["legal_moves"])
                moves.append(legal[seed % len(legal)])
            require(ply >= 8 and len(numerators) > 1, "insufficient position/output variation in golden walk")
            return {"positions": ply + 1, "visited_feature_indices": len(seen),
                    "distinct_numerators": len(numerators), "moves": moves,
                    "accumulator_implementation": active.get("accumulator_implementation")}

    for game in GAMES:
        run("golden/quantized-inference-legal-walk/" + game, lambda game=game: golden_walk(game))

    manual = {
        "chess": ("4k3/8/2n5/3P4/8/8/8/4K3 w - - 0 1",
                  {(4, 0, 0, 63), (4, 7, 1, 63), (3, 4, 0, 1), (2, 5, 1, 2)}, set()),
        "shogi": ("8k/3+P5/9/9/4S4/9/9/9/4K4[PPn] w - - 0 1",
                  {(8, 8, 1, 63), (4, 0, 0, 63), (3, 7, 0, 22), (4, 4, 0, 9)},
                  {(0, 19, 2), (1, 21, 1)}),
        "janggi": ("4k4/9/9/9/4p4/9/9/9/9/3K5 w - - 0 1",
                   {(4, 9, 1, 63), (3, 0, 0, 63), (4, 5, 1, 29)}, set()),
        "xiangqi": ("4k4/9/9/9/4p4/9/9/9/9/3K5 w - - 0 1",
                    {(4, 9, 1, 63), (3, 0, 0, 63), (4, 5, 1, 29)}, set()),
        "makruk": ("7k/8/8/3P4/4S3/8/8/4K3 w - - 0 1",
                   {(7, 7, 1, 63), (4, 0, 0, 63), (3, 4, 0, 1), (4, 3, 0, 9)}, set()),
    }

    def manual_mapping(variant: str) -> dict:
        fen, expected_pieces, expected_pockets = manual[variant]
        with Engine(args.engine, timeout=args.timeout) as e:
            e.variant(variant)
            e.position("fen " + fen)
            activate(e, model_path(variant))
            state = features(e)
            require(semantic_pieces(state) == expected_pieces, "semantic native piece mapping differs")
            require({(p["color"], p["type"], p["count"]) for p in state["pockets"]} == expected_pockets,
                    "native pocket kind/count/color differs")
            if variant == "shogi":
                promoted = next(p for p in state["pieces"] if p["file"] == 3)
                require(promoted["promoted"] and promoted["unpromoted_type"] == 19,
                        "promoted Shogi pawn lost original native identity")
            check_prediction(e, models[variant + "-a"], state)
            return {"fen": state["fen"], "features": state["features"]}

    for game in GAMES:
        run("features/manual-native-identities-and-board-stride/" + game,
            lambda game=game: manual_mapping(game))

    def shogi_sfen_perspective() -> dict:
        with Engine(args.engine, timeout=args.timeout) as e:
            e.variant("shogi")
            # SFEN's 'b' denotes sente, internally WHITE. USI column numbering
            # must not leak into the absolute-board feature coordinate system.
            e.position("sfen 8k/3+P5/9/9/4S4/9/9/9/4K4 b 2Pn 1")
            activate(e, model_path("shogi"))
            state = features(e)
            require(state["side_to_move"] == 0, "SFEN sente/color inversion")
            require(semantic_pieces(state) == manual["shogi"][1], "SFEN board orientation differs")
            require({(p["color"], p["type"], p["count"]) for p in state["pockets"]} == manual["shogi"][2],
                    "SFEN hand/count interpretation differs")
            check_prediction(e, models["shogi-a"], state)
            e.position("sfen 8k/3+P5/9/9/4S4/9/9/9/4K4 w 2Pn 1")
            state = features(e)
            require(state["side_to_move"] == 1, "SFEN gote/color inversion")
            check_prediction(e, models["shogi-a"], state)
            return {"heads_checked": 2, "sfen_hand_count": 2}

    run("features/shogi-sfen-colors-hands-and-both-output-heads", shogi_sfen_perspective)

    def arithmetic(mode: str) -> dict:
        with Engine(args.engine, timeout=args.timeout) as e:
            e.position()
            activate(e, model_path("chess", mode))
            white = check_prediction(e, models["chess-" + mode])
            e.position(moves=["e2e4"])
            black = check_prediction(e, models["chess-" + mode])
            expected = {"rounding": (1, -1), "below-half": (0, 0),
                        "saturation": (10000, -10000), "extreme": (10000, -10000)}[mode]
            require((white["white_cp"], black["white_cp"]) == expected, "signed rounding/clamping contract")
            if mode == "extreme":
                require(white["numerator"] > 2**31 - 1 and black["numerator"] < -(2**31),
                        "overflow fixture failed to exceed 32-bit dot-product range")
            return {"white_head_numerator": white["numerator"], "black_head_numerator": black["numerator"],
                    "white_head_cp": white["white_cp"], "black_head_cp": black["white_cp"]}

    for mode in ("rounding", "below-half", "saturation", "extreme"):
        run("arithmetic/" + mode, lambda mode=mode: arithmetic(mode))

    invalid = mutated_files(model_path("chess").read_bytes(), report["descriptors"]["chess"]["input_dim"])
    for name, content in invalid.items():
        (args.fixture_dir / ("invalid-" + name + ".svnnue")).write_bytes(content)

    def invalid_load(name: str) -> dict:
        with Engine(args.engine, timeout=args.timeout) as e:
            e.position(moves=["e2e4", "e7e5", "g1f3"])
            activate(e, model_path("chess"))
            before = check_prediction(e, models["chess-a"])
            path = args.fixture_dir / ("invalid-" + name + ".svnnue")
            output = set_option(e, "EvalFile", path)
            errors = [line for line in output if "ERROR" in line]
            require(errors, "malformed/incompatible network accepted: " + name)
            after = check_prediction(e, models["chess-a"])
            require(after["generation"] == before["generation"], "failed load changed model generation")
            require(after["numerator"] == before["numerator"]
                    and after["accumulator"] == before["accumulator"], "failed load changed live model")
            current = status(e)
            require(current["loaded"] and current["active"] and current["compatible"],
                    "failed load discarded prior good network")
            require(Path(current["file"]) == model_path("chess"), "failed load changed active model path")
            return {"diagnostic": errors[0], "prior_network_preserved": True}

    for name in invalid:
        run("loader/reject-preserving-active-model/" + name, lambda name=name: invalid_load(name))

    def reload_and_auto() -> dict:
        with Engine(args.engine, timeout=args.timeout) as e:
            e.position(moves=["e2e4", "e7e5"])
            activate(e, model_path("chess"))
            before = check_prediction(e, models["chess-a"])
            successful_option(e, "EvalFile", model_path("chess", "b"))
            after = check_prediction(e, models["chess-b"])
            require(after["generation"] != before["generation"], "successful reload did not invalidate cache")
            require(after["accumulator"] != before["accumulator"], "fixture models unexpectedly identical")
            successful_option(e, "EvalBackend", "Classical")
            require(probe(e)["used"] is False, "explicit Classical still used loaded model")
            successful_option(e, "EvalBackend", "Auto")
            check_prediction(e, models["chess-b"])
            e.variant("makruk")
            e.position()
            inactive = probe(e)
            require(inactive["used"] is False and inactive["compatible"] is False,
                    "Auto used chess weights for same-size Makruk board")
            successful_option(e, "EvalFile", model_path("makruk"))
            check_prediction(e, models["makruk-a"])
            return {"successful_reload": True, "auto_compatible_only": True}

    run("activation/reload-invalidation-and-auto-variant-compatibility", reload_and_auto)

    def rejected_variant(variant: str, chess960: bool = False) -> dict:
        with Engine(args.engine, timeout=args.timeout) as e:
            e.position()
            activate(e, model_path("chess"))
            e.variant(variant, chess960)
            e.position()
            inactive = probe(e)
            require(inactive["used"] is False and inactive["compatible"] is False,
                    "incompatible game/mode silently accepted neural weights")
            exported = query(e, "nnuefeatures", "NNUE_FEATURES")
            require("error" in exported, "unsupported feature state silently exported for training")
            if not chess960:
                output = set_option(e, "EvalFile", model_path("chess"))
                require(any("ERROR" in line for line in output), "unsupported native game accepted model")
            legal = e.legal()
            best, _ = e.search("depth 2")
            require(best in legal, "classical fallback returned illegal move")
            return {"fallback_reason": inactive["reason"], "legal_bestmove": best}

    for game in ("atomic", "crazyhouse", "janggitraditional", "minishogi", "fischerandom"):
        run("compatibility/fallback-unsupported/" + game,
            lambda game=game: rejected_variant(game))
    run("compatibility/fallback-chess960-position-mode", lambda: rejected_variant("chess", True))

    def search_uses_model(variant: str) -> dict:
        with Engine(args.engine, timeout=args.timeout) as e:
            e.variant(variant)
            e.position()
            activate(e, model_path(variant))
            root = check_prediction(e, models[variant + "-a"])
            e.option("Threads", 2)
            e.option("MultiPV", 2)
            legal = e.legal()
            best, lines = e.search("depth 3")
            require(best in legal, "NNUE search returned illegal best move")
            require(any(" multipv 2 " in line for line in lines), "NNUE MultiPV did not report second line")
            after = check_prediction(e, models[variant + "-a"])
            require(root["accumulator"] == after["accumulator"], "search mutated root accumulator")
            require(any("NNUE_STATUS" in line and '"active":true' in line for line in lines),
                    "search did not confirm an active compatible NNUE model")
            return {"threads": 2, "multipv": 2, "depth": 3, "bestmove": best,
                    "root_prediction_preserved": True}

    for game in GAMES:
        run("search/active-model-two-threads-multipv/" + game,
            lambda game=game: search_uses_model(game))

    def actual_search_effect() -> dict:
        signatures = []
        for suffix in ("a", "b"):
            with Engine(args.engine, timeout=args.timeout) as e:
                e.option("Threads", 1)
                e.option("StocklessHybrid", False)
                e.option("StocklessOverdrive", False)
                e.position()
                activate(e, model_path("chess", suffix))
                _, lines = e.search("depth 2")
                scores = [re.search(r"\bscore cp (-?\d+)\b", line) for line in lines if " depth 2 " in line]
                scores = [int(match.group(1)) for match in scores if match]
                require(scores, "missing complete neural search score")
                signatures.append(scores[-1])
        require(signatures[0] != signatures[1], "changing actual network had no visible search score effect")
        return {"network_a_score_cp": signatures[0], "network_b_score_cp": signatures[1]}

    run("search/changing-network-changes-evaluation", actual_search_effect)

    if args.cpp_test:
        def cpp_harness() -> dict:
            process = subprocess.run([str(args.cpp_test.resolve()), str(args.fixture_dir)],
                                     cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     timeout=max(args.timeout, 120))
            log = args.fixture_dir / "cpp-nnue-test.log"
            log.write_text(process.stdout, encoding="utf-8")
            result_lines = [line for line in process.stdout.splitlines() if line.startswith("NNUE_TEST_RESULT ")]
            require(process.returncode == 0 and len(result_lines) == 1,
                    f"C++ accumulator harness failed ({process.returncode}); log {log}\n{process.stdout[-6000:]}")
            result = json.loads(result_lines[0].split(" ", 1)[1])
            require(result.get("failures") == 0 and result.get("transitions", 0) >= 200
                    and result.get("incremental_probes", 0) >= 400,
                    "C++ harness reported insufficient coverage or failures")
            result["log"] = str(log)
            return result
        run("cpp/incremental-special-moves-undo-reload-and-isolation", cpp_harness)
    else:
        report["limitations"].append("C++ make/undo accumulator harness not requested in this run")
    return finish(args.report, report, started)


def finish(path: Path, report: dict, started: float) -> int:
    passed = sum(bool(item["passed"]) for item in report["tests"])
    failed = len(report["tests"]) - passed
    report.update(passed=passed, failed=failed, elapsed_seconds=round(time.monotonic() - started, 3))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"NNUE_VALIDATION_RESULT passed={passed} failed={failed} report={path}", flush=True)
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
