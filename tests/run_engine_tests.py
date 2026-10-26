#!/usr/bin/env python3
"""Black-box rule and protocol regressions against the actual engine binary.

The perft oracle is the pinned Fairy-Stockfish test suite. It proves agreement
with that revision, not independent certification of regional rule conventions.
No optional dependency (python-chess, expect, pyffish) is required.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import platform
from queue import Empty, Queue
import re
import shlex
import subprocess
import sys
import threading
import time
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
PERFT_SOURCE = ROOT / "upstream/fairy-stockfish/tests/perft.sh"
MAIN_VARIANTS = [
    "chess", "fischerandom", "nocastle", "3check", "5check", "kingofthehill",
    "racingkings", "horde", "atomic", "crazyhouse", "antichess", "giveaway",
    "seirawan", "capablanca", "gothic", "grand", "embassy", "janus", "duck",
    "torpedo", "pawnsideways", "shogi", "minishogi", "kyotoshogi", "dobutsu",
    "torishogi", "yarishogi", "okisakishogi", "janggi", "janggitraditional",
    "janggimodern", "janggicasual", "xiangqi", "minixiangqi", "manchu", "supply",
    "makruk", "makpong", "asean", "cambodian", "karouk", "sittuyin", "shatranj",
    "courier", "shatar", "jesonmor", "chaturanga",
]
PLAY_VARIANTS = ["chess", "atomic", "crazyhouse", "shogi", "janggi", "xiangqi", "makruk", "antichess"]


class Engine:
    def __init__(self, executable: Path, protocol: str = "uci", timeout: float = 30):
        self.executable = executable.resolve()
        self.timeout = timeout
        self.process = subprocess.Popen(
            [str(self.executable), "noautoload"], cwd=self.executable.parent,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        self.queue: Queue[str | None] = Queue()
        self.recent: deque[str] = deque(maxlen=40)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.protocol = protocol
        self.send(protocol)
        if protocol == "xboard":
            self.send("protover 2")
            self.handshake = self.until(lambda s: s == "feature done=1")
        else:
            self.handshake = self.until(lambda s: s == protocol + "ok")
            self.send("setoption name Use NNUE value false")
            self.ready()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.queue.put(line.rstrip("\r\n"))
        self.queue.put(None)

    def send(self, command: str) -> None:
        if self.process.poll() is not None:
            raise AssertionError(f"engine exited {self.process.returncode}: {list(self.recent)}")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except (OSError, BrokenPipeError) as exc:
            raise AssertionError(f"engine input failed: {list(self.recent)}") from exc

    def until(self, predicate: Callable[[str], bool], timeout: float | None = None) -> list[str]:
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        lines: list[str] = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AssertionError(f"engine response timeout; recent output: {list(self.recent)}")
            try:
                line = self.queue.get(timeout=remaining)
            except Empty as exc:
                raise AssertionError(f"engine response timeout; recent output: {list(self.recent)}") from exc
            if line is None:
                raise AssertionError(f"unexpected engine EOF (exit {self.process.poll()}): {list(self.recent)}")
            self.recent.append(line)
            lines.append(line)
            if predicate(line):
                return lines

    def ready(self) -> list[str]:
        self.send("isready")
        return self.until(lambda s: s == "readyok")

    def option(self, name: str, value: str | int | bool) -> None:
        if isinstance(value, bool):
            value = str(value).lower()
        self.send(f"setoption name {name} value {value}")
        self.ready()

    def variant(self, name: str, chess960: bool = False) -> None:
        self.option("UCI_Chess960", chess960)
        self.option("UCI_Variant", name)
        self.send("ucinewgame")
        self.ready()

    def position(self, position: str = "startpos", moves: list[str] | None = None) -> None:
        self.send("position " + position + (" moves " + " ".join(moves) if moves else ""))

    def perft(self, depth: int, timeout: float | None = None) -> tuple[int, dict[str, int]]:
        self.send(f"go perft {depth}")
        lines = self.until(lambda s: bool(re.match(r"Nodes searched\s*:\s*\d+", s)), timeout)
        count = int(re.search(r"(\d+)\s*$", lines[-1]).group(1))
        divide = {}
        for line in lines:
            match = re.match(r"^(\S+):\s+(\d+)\s*$", line)
            if match:
                divide[match.group(1)] = int(match.group(2))
        return count, divide

    def legal(self) -> set[str]:
        count, divide = self.perft(1)
        if count != len(divide):
            raise AssertionError(f"perft(1) divide/count mismatch: {count} vs {len(divide)}")
        return set(divide)

    def search(self, limit: str = "depth 3", timeout: float | None = None) -> tuple[str, list[str]]:
        self.send("go " + limit)
        lines = self.until(lambda s: s.startswith("bestmove "), timeout)
        return lines[-1].split()[1], lines

    def dump(self) -> list[str]:
        self.send("d")
        return self.ready()

    def close(self, exception: bool = False) -> None:
        if self.process.poll() is None:
            try:
                self.send("quit")
                self.process.wait(timeout=5)
            except (subprocess.TimeoutExpired, AssertionError, OSError):
                self.process.kill()
                self.process.wait(timeout=5)
                if not exception:
                    raise AssertionError("engine failed to terminate cleanly")
        if self.process.returncode != 0 and not exception:
            raise AssertionError(f"engine exited {self.process.returncode}")
        if self.process.stdin:
            self.process.stdin.close()
        if self.process.stdout:
            self.process.stdout.close()

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, kind, value, tb) -> None:
        self.close(exception=kind is not None)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perft_cases(quick: bool, full: bool) -> list[dict]:
    result = []
    for line in PERFT_SOURCE.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("expect perft.exp "):
            continue
        words = shlex.split(line, comments=True)
        variant, position = words[2:4]
        depth, expected = map(int, words[4:6])
        if not full and expected > 5_000_000:
            continue
        if quick and (position != "startpos" or variant not in MAIN_VARIANTS):
            continue
        result.append({"variant": variant, "position": position, "depth": depth,
                       "expected": expected, "chess960": words[6] == "true"})
    if not result:
        raise AssertionError("no upstream perft cases extracted")
    return result


def search_signature(engine: Engine, depth: int = 4) -> dict:
    """Deterministic depth-limited output, excluding elapsed time and NPS."""
    best, lines = engine.search(f"depth {depth}")
    signature = None
    for line in lines:
        actual_depth = re.search(r"\bdepth (\d+)\b", line)
        score = re.search(r"\bscore (cp|mate) (-?\d+)\b", line)
        nodes = re.search(r"\bnodes (\d+)\b", line)
        if actual_depth and int(actual_depth.group(1)) == depth and score and nodes:
            signature = {"bestmove": best, "depth": depth,
                         "score": {"unit": score.group(1), "value": int(score.group(2))},
                         "nodes": int(nodes.group(1))}
    if signature is None:
        raise AssertionError(f"search did not complete depth {depth} with score and node information")
    return signature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--reference", type=Path, help="unmodified pinned FSF binary for legal-set comparison")
    parser.add_argument("--report", type=Path, default=ROOT / "build/test-report.json")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--full-perft", action="store_true", help="also execute upstream cases exceeding 5 million nodes")
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    started = time.monotonic()
    report = {
        "schema": 1, "engine": str(args.engine.resolve()), "engine_sha256": sha256(args.engine),
        "reference_sha256": sha256(args.reference) if args.reference else None,
        "platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
        "threads_default": 1, "evaluation": "classical (Use NNUE=false)",
        "quick": args.quick, "full_perft": args.full_perft,
        "oracle": "pinned Fairy-Stockfish tests/perft.sh and optional pinned executable",
        "limitations": ["bounded regression coverage; no Elo estimate", "no long-match or race-detector certification",
                        "matching FSF is not independent validation of every regional/Chess.com rule"],
        "tests": [],
    }

    def run(name: str, action: Callable[[], dict | None]) -> None:
        before = time.monotonic()
        try:
            evidence = action() or {}
            item = {"name": name, "passed": True, "evidence": evidence}
        except Exception as exc:
            item = {"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
        item["elapsed_seconds"] = round(time.monotonic() - before, 3)
        report["tests"].append(item)
        print(("PASS " if item["passed"] else "FAIL ") + name, flush=True)
        if not item["passed"]:
            print(item["error"], flush=True)

    def protocol_test(protocol: str, expected_variant: str) -> dict:
        with Engine(args.engine, protocol, args.timeout) as e:
            handshake = "\n".join(e.handshake)
            if f"default {expected_variant}" not in handshake:
                raise AssertionError(f"{protocol} did not advertise default {expected_variant}")
            e.position()
            legal = e.legal()
            best, lines = e.search("depth 2")
            if best not in legal:
                raise AssertionError(f"{protocol} illegal bestmove: {best}")
            return {"default_variant": expected_variant, "legal_moves": len(legal), "bestmove": best}

    for proto, variant in [("uci", "chess"), ("usi", "shogi"), ("ucci", "xiangqi")]:
        run("protocol/" + proto, lambda p=proto, v=variant: protocol_test(p, v))

    def xboard_test() -> dict:
        with Engine(args.engine, "xboard", args.timeout) as e:
            e.send("new")
            e.send("force")
            e.send("ping 173")
            e.until(lambda s: s == "pong 173")
            e.send("sd 2")
            e.send("go")
            lines = e.until(lambda s: s.startswith("move "))
            move = lines[-1].split()[1]
            with Engine(args.engine) as uci:
                uci.position()
                if move not in uci.legal():
                    raise AssertionError(f"XBoard illegal bestmove: {move}")
            return {"ping_pong": True, "bestmove": move}
    run("protocol/xboard", xboard_test)

    with Engine(args.engine, timeout=args.timeout) as engine:
        report["identity"] = [s for s in engine.handshake if s.startswith("id ")]
        variant_line = next(s for s in engine.handshake if "UCI_Variant" in s and " var " in s)
        available = set(variant_line.split(" var ")[1:])
        report["advertised_variants"] = sorted(available)

        for case in perft_cases(args.quick, args.full_perft):
            def perft_test(c=case) -> dict:
                if c["variant"] not in available:
                    raise AssertionError("required native variant is not compiled in")
                engine.variant(c["variant"], c["chess960"])
                engine.position(c["position"])
                actual, _ = engine.perft(c["depth"], max(args.timeout, 180 if args.full_perft else args.timeout))
                if actual != c["expected"]:
                    raise AssertionError(f"expected {c['expected']}, actual {actual}, position {c['position']}")
                return {**c, "actual": actual}
            label = case["variant"] + "/depth-" + str(case["depth"]) + "/" + str(len(report["tests"]))
            run("perft/" + label, perft_test)

        smoke_variants = ["chess", "shogi", "janggi", "xiangqi", "atomic", "crazyhouse", "duck", "antichess"] if args.quick else MAIN_VARIANTS
        for variant in smoke_variants:
            def smoke(v=variant) -> dict:
                if v not in available:
                    raise AssertionError(f"requested smoke variant absent: {v}")
                engine.variant(v)
                engine.position()
                legal = engine.legal()
                best, lines = engine.search("depth 2")
                if best not in legal:
                    raise AssertionError(f"illegal bestmove {best} for {v}")
                return {"variant": v, "legal_moves": len(legal), "bestmove": best}
            run("search/" + variant, smoke)

        def limits_test() -> dict:
            engine.variant("chess")
            engine.option("Hash", 32)
            engine.option("MultiPV", 3)
            engine.position()
            legal = engine.legal()
            best, lines = engine.search("depth 4")
            if best not in legal:
                raise AssertionError("MultiPV bestmove illegal")
            pv_ids = {int(m.group(1)) for s in lines if (m := re.search(r"\bmultipv (\d+)", s))}
            if not {1, 2, 3}.issubset(pv_ids):
                raise AssertionError(f"incomplete MultiPV output {pv_ids}")
            engine.option("MultiPV", 1)
            data = {"multipv": sorted(pv_ids)}
            for limit in ["nodes 1500", "movetime 60", "wtime 1500 btime 1500 winc 10 binc 10 movestogo 20"]:
                engine.position()
                start = time.monotonic()
                best, _ = engine.search(limit, timeout=10)
                if best not in legal:
                    raise AssertionError(f"illegal bestmove with {limit}")
                elapsed = time.monotonic() - start
                if elapsed > 5:
                    raise AssertionError(f"{limit} did not respect a bounded response: {elapsed}")
                data[limit] = {"elapsed_seconds": round(elapsed, 3), "bestmove": best}
            return data
        run("protocol/limits-and-multipv", limits_test)

        def stop_test() -> dict:
            engine.variant("chess")
            engine.position()
            legal = engine.legal()
            engine.send("go infinite")
            engine.until(lambda s: bool(re.search(r"\bdepth [2-9]\b", s)))
            start = time.monotonic()
            engine.send("stop")
            lines = engine.until(lambda s: s.startswith("bestmove "), 5)
            if lines[-1].split()[1] not in legal:
                raise AssertionError("stop returned illegal move")
            elapsed = time.monotonic() - start
            engine.option("Ponder", True)
            engine.position()
            engine.send("go ponder wtime 2000 btime 2000")
            engine.until(lambda s: bool(re.search(r"\bdepth [2-9]\b", s)))
            engine.send("ponderhit")
            lines = engine.until(lambda s: s.startswith("bestmove "), 10)
            if lines[-1].split()[1] not in legal:
                raise AssertionError("ponderhit returned illegal move")
            engine.option("Ponder", False)
            return {"stop_seconds": round(elapsed, 4), "ponderhit": True}
        run("protocol/stop-and-ponder", stop_test)

        def multithread_test() -> dict:
            engine.option("Threads", 2)
            for v in ["chess", "shogi", "janggi", "xiangqi", "crazyhouse"]:
                engine.variant(v)
                engine.position()
                legal = engine.legal()
                best, _ = engine.search("nodes 2500")
                if best not in legal:
                    raise AssertionError(f"two-thread illegal bestmove for {v}")
            engine.option("Threads", 1)
            return {"threads": 2, "variants": 5, "scope": "bounded concurrent search, not race-detector proof"}
        run("search/two-threads", multithread_test)

        def hybrid_test() -> dict:
            outputs = {}
            for name, enabled, mobile in [("off", False, False), ("desktop", True, False), ("mobile", True, True)]:
                engine.option("StocklessHybrid", enabled)
                engine.option("StocklessMobile", mobile)
                engine.variant("chess")
                engine.position("fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                legal = engine.legal()
                best, lines = engine.search("depth 6")
                if best not in legal:
                    raise AssertionError("hybrid mode produced illegal move")
                nodes = [int(m.group(1)) for s in lines if (m := re.search(r"\bnodes (\d+)", s))]
                if not nodes:
                    raise AssertionError("search did not report node counts")
                outputs[name] = {"nodes": nodes[-1], "bestmove": best}
            engine.option("StocklessMobile", False)
            if outputs["desktop"]["nodes"] == outputs["off"]["nodes"]:
                raise AssertionError("desktop hybrid ON/OFF produced identical node count on regression position; verify real integration")
            if outputs["mobile"]["nodes"] == outputs["off"]["nodes"]:
                raise AssertionError("mobile hybrid ON/OFF produced identical node count; fractional policy disappeared")
            return {"modes": outputs, "interpretation": "demonstrates real search effect, not playing-strength improvement"}
        run("search/hybrid-has-real-effect", hybrid_test)

        def overdrive_test() -> dict:
            evidence = {}
            for mobile in [False, True]:
                for enabled in [False, True]:
                    engine.option("StocklessMobile", mobile)
                    engine.option("StocklessOverdrive", enabled)
                    engine.variant("chess")
                    engine.position("fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                    legal = engine.legal()
                    best, lines = engine.search("depth 7", timeout=15)
                    if best not in legal:
                        raise AssertionError("overdrive returned illegal move")
                    nodes = [int(m.group(1)) for s in lines if (m := re.search(r"\bnodes (\d+)", s))]
                    if not nodes:
                        raise AssertionError("overdrive search did not complete a reported depth")
                    evidence[f"mobile={mobile},overdrive={enabled}"] = {"bestmove": best, "nodes": nodes[-1]}
            if evidence["mobile=True,overdrive=False"] != evidence["mobile=True,overdrive=True"]:
                raise AssertionError("desktop Overdrive leaked into mobile search")
            engine.option("StocklessMobile", False)
            engine.option("StocklessOverdrive", False)
            return {"modes": evidence, "scope": "bounded forcing-position regression, not exhaustive depth proof"}
        run("search/overdrive-bounded-and-mobile-inert", overdrive_test)

        plies = 8 if args.quick else 32
        for variant in PLAY_VARIANTS:
            def playout(v=variant) -> dict:
                engine.variant(v)
                history: list[str] = []
                terminated = False
                for _ in range(plies):
                    engine.position(moves=history)
                    legal = engine.legal()
                    best, lines = engine.search("depth 2")
                    if best in {"(none)", "0000"}:
                        # Terminal-by-rule positions can retain geometric moves.
                        if legal and not any(re.search(r"\bscore mate 0\b", s) for s in lines):
                            raise AssertionError("no bestmove with nonterminal legal set")
                        terminated = True
                        break
                    if best not in legal:
                        raise AssertionError(f"illegal move at ply {len(history)}: {best}")
                    history.append(best)
                return {"variant": v, "plies": len(history), "ended": terminated, "moves": history}
            run("playout/" + variant, playout)

        if args.reference:
            with Engine(args.reference, timeout=args.timeout) as reference:
                for variant in sorted(available):
                    def compare(v=variant) -> dict:
                        engine.variant(v)
                        reference.variant(v)
                        engine.position()
                        reference.position()
                        ours, theirs = engine.legal(), reference.legal()
                        if ours != theirs:
                            raise AssertionError(f"legal sets differ: ours-only={sorted(ours-theirs)[:12]} reference-only={sorted(theirs-ours)[:12]}")
                        return {"variant": v, "legal_moves": len(ours)}
                    run("differential/startpos/" + variant, compare)
                for variant in PLAY_VARIANTS:
                    def compare_line(v=variant) -> dict:
                        engine.variant(v)
                        reference.variant(v)
                        moves: list[str] = []
                        for ply in range(12):
                            engine.position(moves=moves)
                            reference.position(moves=moves)
                            ours, theirs = engine.legal(), reference.legal()
                            if ours != theirs:
                                raise AssertionError(f"legal sets differ at ply {ply} after {' '.join(moves)}")
                            if not ours:
                                break
                            moves.append(sorted(ours)[(17 * ply + 11) % len(ours)])
                        return {"variant": v, "plies": len(moves), "moves": moves}
                    run("differential/playout/" + variant, compare_line)

                # A policy-off port must preserve the native search where rules
                # were unchanged. Intentional Ataxx/Seirawan/Shouse hash repairs
                # are excluded; legal-set comparison above still covers them.
                # Use fresh processes for each fixture: the unmodified upstream
                # engine retains variant-dependent pawn/material caches across
                # variant switches even after ucinewgame. TT/history clearing
                # alone would compare against a polluted reference evaluation.
                baseline_positions = [
                    ("chess", "startpos", False),
                    ("fischerandom", "fen bqrbkrnn/pppppppp/8/8/8/8/PPPPPPPP/BQRBKRNN w CFcf - 0 1", True),
                    ("atomic", "startpos", False),
                    ("shogi", "startpos", False),
                    ("janggi", "startpos", False),
                    ("xiangqi", "startpos", False),
                    ("makruk", "startpos", False),
                ]
                for variant, position, chess960 in baseline_positions:
                    def compare_search(v=variant, p=position, frc=chess960) -> dict:
                        with Engine(args.engine, timeout=args.timeout) as candidate, \
                             Engine(args.reference, timeout=args.timeout) as native:
                            candidate.option("StocklessHybrid", False)
                            for e in (candidate, native):
                                e.option("Threads", 1)
                                e.option("Hash", 16)
                                e.option("MultiPV", 1)
                                e.option("Ponder", False)
                                e.option("Skill Level", 20)
                                e.variant(v, frc)
                                e.position(p)
                            ours = search_signature(candidate)
                            theirs = search_signature(native)
                        if ours != theirs:
                            raise AssertionError(f"Hybrid=false differs from native FSF: ours={ours}, reference={theirs}")
                        return {"variant": v, "position": p, "chess960": frc,
                                "threads": 1, "hash_mb": 16, "hybrid": False, "process_isolation": True,
                                "actual": ours, "reference": theirs,
                                "scope": "exact equality for this depth-limited fixture, not universal equivalence"}
                    run("differential/hybrid-off-search/" + variant, compare_search)

    report["passed"] = sum(bool(t["passed"]) for t in report["tests"])
    report["failed"] = len(report["tests"]) - report["passed"]
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"RESULT {report['passed']} passed, {report['failed']} failed; {args.report}", flush=True)
    return int(report["failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
