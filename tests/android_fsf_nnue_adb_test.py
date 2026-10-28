#!/usr/bin/env python3
"""Check original Fairy-Stockfish .nnue evaluation on an ARM64 Android device.

Requires a real arm64-v8a Android device or emulator visible to adb. Merely
loading a file or falling back to classical evaluation does not count as PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cmd(args: list[str], *, timeout: int = 180, input_text: str | None = None) -> str:
    proc = subprocess.run(
        args, input=input_text, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        encoding="utf-8", errors="replace", timeout=timeout, check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"Exit {proc.returncode}: {args!r}\n{proc.stdout[-3500:]}")
    return proc.stdout.replace("\r", "")


def require_original_hash(path: Path) -> str:
    expected = path.stem.rsplit("-", 1)[-1]
    if not re.fullmatch(r"[0-9a-f]{12}", expected):
        raise ValueError(f"Missing official 12-digit filename hash: {path.name}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith(expected):
        raise ValueError(f"NNUE SHA256 mismatch: {path.name} actual={digest}")
    return digest


def parse_native_fsf_evaluation(output: str, filename: str) -> tuple[str, str]:
    if "info string ERROR" in output or "classical evaluation enabled" in output:
        raise AssertionError(f"Native FSF NNUE failed or fell back: {output[-2500:]}")
    loaded = [line.strip() for line in output.splitlines()
              if "NNUE evaluation using " in line and filename in line and "enabled" in line]
    final = [line.strip() for line in output.splitlines()
             if line.strip().startswith("Final evaluation")]
    if not loaded:
        raise AssertionError(f"No selected native FSF network {filename}: {output[-2500:]}")
    if not final:
        raise AssertionError(f"No computed final evaluation: {output[-2500:]}")
    if not re.search(r"Final evaluation\s+[-+]?\d+(?:\.\d+)?", final[-1]):
        raise AssertionError(f"Final evaluation was nonnumeric: {final[-1]}")
    return loaded[-1], final[-1]


def run(args: argparse.Namespace) -> int:
    catalog = json.loads((ROOT / "config/fsf-networks.json").read_text(encoding="utf-8"))
    records = list(catalog["networks"])
    root = args.net_dir.resolve()
    local_dir = root / "networks" if (root / "networks").is_dir() else root
    engine = args.engine.resolve()
    if not engine.is_file():
        raise FileNotFoundError(engine)
    if not local_dir.is_dir():
        raise FileNotFoundError(local_dir)

    for item in records:
        network = local_dir / item["filename"]
        if not network.is_file():
            raise FileNotFoundError(network)
        item["sha256"] = require_original_hash(network)

    adb = args.adb
    remote = args.remote_dir
    abi = cmd([adb, "shell", "getprop", "ro.product.cpu.abi"]).strip()
    machine = cmd([adb, "shell", "uname", "-m"]).strip()
    print(f"ANDROID_ENV abi={abi} machine={machine}", flush=True)
    if abi != "arm64-v8a" or machine not in {"aarch64", "arm64"}:
        raise RuntimeError(f"Not a native ARM64 Android environment: abi={abi}, machine={machine}")

    cmd([adb, "shell", "mkdir", "-p", f"{remote}/networks"])
    cmd([adb, "push", str(engine), f"{remote}/stockless-variants"], timeout=240)
    cmd([adb, "shell", "chmod", "755", f"{remote}/stockless-variants"])
    for item in records:
        cmd([adb, "push", str(local_dir / item["filename"]),
             f"{remote}/networks/{item['filename']}"], timeout=240)

    trials = [(item["variant"], item["filename"]) for item in records]
    if args.janggi_modes:
        janggi_name = next(x["filename"] for x in records if x["variant"] == "janggi")
        trials += [(name, janggi_name) for name in
                   ("janggimodern", "janggitraditional", "janggicasual")]

    results = []
    ok = True
    for variant, filename in trials:
        script = (f"uci\nsetoption name UCI_Variant value {variant}\n"
                  "isready\nposition startpos\neval\nquit\n")
        start = time.monotonic()
        try:
            output = cmd([adb, "shell", "-T",
                          f"cd {shlex.quote(remote)} && ./stockless-variants"],
                         input_text=script, timeout=args.timeout)
            loaded, final = parse_native_fsf_evaluation(output, filename)
            result = {"variant": variant, "network": filename, "passed": True,
                      "nnue_line": loaded, "evaluation_line": final,
                      "duration_seconds": round(time.monotonic() - start, 2)}
            print(f"ANDROID_FSF_NNUE_PASS {variant}: {loaded} | {final}", flush=True)
        except (AssertionError, RuntimeError, subprocess.TimeoutExpired) as exc:
            result = {"variant": variant, "network": filename, "passed": False,
                      "error": str(exc),
                      "duration_seconds": round(time.monotonic() - start, 2)}
            print(f"ANDROID_FSF_NNUE_FAIL {variant}: {exc}", flush=True)
            ok = False
        results.append(result)

    report = {"schema": 1, "test": "actual ARM64 Android FSF NNUE evaluation",
              "android_abi": abi, "uname_m": machine,
              "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
              "passed": sum(int(r["passed"]) for r in results),
              "failed": sum(int(not r["passed"]) for r in results),
              "results": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ANDROID_FSF_NNUE_RESULT passed={report['passed']} failed={report['failed']}", flush=True)
    return 0 if ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--net-dir", required=True, type=Path,
                        help="Extracted bundle folder or its networks/ subfolder")
    parser.add_argument("--report", default=Path("build/android-fsf-nnue-report.json"), type=Path)
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--remote-dir", default="/data/local/tmp/stockless-nnue-test")
    parser.add_argument("--timeout", default=180, type=int)
    parser.add_argument("--janggi-modes", action="store_true")
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
