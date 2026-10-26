#!/usr/bin/env python3
"""Link actual rule tests against exactly the objects used by the engine."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "build/engine/src")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--arch", choices=["x86-64", "x86-64-avx2"], default="x86-64")
    parser.add_argument("--extra-flag", action="append", default=[])
    args = parser.parse_args()
    output = args.source.resolve().parent / "tests"
    command = [sys.executable, str(ROOT / "tests/build_cpp_tests.py"),
               "--source", str(args.source), "--out", str(output),
               "--cxx", args.cxx, "--arch", args.arch]
    command.extend("--extra-flag=" + flag for flag in args.extra_flag)
    subprocess.run(command, check=True)
    for name in ("rules_test", "policy_test", "stockless_policy_regression"):
        subprocess.run([str(output / name)], check=True)


if __name__ == "__main__":
    main()
