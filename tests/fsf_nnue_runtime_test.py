#!/usr/bin/env python3
"""Run genuine FSF .nnue files through the compiled engine evaluation path."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from fsf_networks import verify


def invoke(engine: Path, commands: str, timeout: int = 75) -> str:
    proc = subprocess.run(
        [str(engine)], input=commands, text=True,
        encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout, cwd=engine.parent,
    )
    if proc.returncode:
        raise RuntimeError(f"Engine exited {proc.returncode}: {proc.stdout[-3000:]}")
    if "info string ERROR" in proc.stdout or "info string ERROR:" in proc.stdout:
        raise RuntimeError(f"Engine reported an error: {proc.stdout[-3000:]}")
    return proc.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, type=Path)
    parser.add_argument("--fsf-net-dir", type=Path, default=ROOT / "build/fsf-networks")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    engine = args.engine.resolve()
    if not engine.is_file():
        raise SystemExit(f"Engine missing: {engine}")
    records = verify(args.fsf_net_dir.resolve())
    installed = engine.parent / "networks"
    installed.mkdir(parents=True, exist_ok=True)
    # Test the exact folder layout users get on extraction; zero-byte stubs
    # cannot pass the source hash and the C++ NNUE evaluation tests.
    for record in records:
        shutil.copy2(args.fsf_net_dir.resolve() / "networks" / record["filename"],
                     installed / record["filename"])
    handshake = invoke(engine, "uci\nisready\nquit\n")
    if "option name Use NNUE type check default true" not in handshake:
        raise AssertionError("Extracted bundle did not enable FSF NNUE by default")
    for record in records:
        if record["filename"] not in handshake:
            raise AssertionError("EvalFile default does not include " + record["filename"])
    passed = 1
    for record in records:
        variant, filename = record["variant"], record["filename"]
        output = invoke(
            engine,
            f"uci\nsetoption name UCI_Variant value {variant}\n"
            "position startpos\neval\nisready\nquit\n",
        )
        if "NNUE evaluation using" not in output or filename not in output:
            raise AssertionError(f"FSF NNUE did not activate for {variant}: {output[-3000:]}")
        if "NNUE evaluation" not in output or "Final evaluation" not in output:
            raise AssertionError(f"FSF NNUE did not evaluate {variant}: {output[-3000:]}")
        passed += 1
        print(f"FSF_NATIVE_NNUE_PASS {variant} {filename}", flush=True)

    report = {
        "schema": 1,
        "suite": "FSF original network loader, per-variant and bundle auto-selection",
        "passed": passed, "failed": 0,
        "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        "networks": [{k: n[k] for k in ("variant", "filename", "sha256")} for n in records],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"FSF_NNUE_VALIDATION_RESULT passed={passed} failed=0")


if __name__ == "__main__":
    main()
