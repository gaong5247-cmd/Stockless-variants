#!/usr/bin/env python3
"""Link rule/policy harnesses against a previously built all-variant engine.

Run after the normal ARCH=x86-64 largeboards=yes all=yes nnue=no build.
The supplied engine objects must use the same board-size/SIMD compile options.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "build/engine/src")
    parser.add_argument("--out", type=Path, default=ROOT / "build/tests")
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--arch", choices=["x86-64", "x86-64-avx2"], default="x86-64")
    parser.add_argument("--extra-flag", action="append", default=[])
    parser.add_argument("--reference", action="store_true", help="build only rules, no Stockless policy")
    args = parser.parse_args()
    src = args.source.resolve()
    objects = [str(p) for p in sorted(src.glob("*.o")) if p.name != "main.o"]
    if len(objects) < 20:
        parser.error("compile the engine first; expected its complete object files")
    args.out.mkdir(parents=True, exist_ok=True)
    # FSF's bit-operator helpers type-pun enum/int references. Match its
    # -fno-strict-aliasing contract also in this TU's inline attack helpers.
    flags = [args.cxx, "-std=c++17", "-O2", "-fno-strict-aliasing", "-DLARGEBOARDS", "-DALLVARS",
             "-DPRECOMPUTED_MAGICS", "-DNNUE_EMBEDDING_OFF", "-DIS_64BIT",
             "-DUSE_PTHREADS", "-DUSE_SSE2", "-msse2", "-I" + str(src)]
    if args.arch == "x86-64-avx2":
        flags += ["-DUSE_POPCNT", "-DUSE_AVX2", "-DUSE_SSE41", "-DUSE_SSSE3",
                  "-msse3", "-mpopcnt", "-mavx2", "-mbmi", "-msse4.1", "-mssse3"]
    flags += args.extra_flag
    for name in ["rules_test"] + ([] if args.reference else ["policy_test", "nnue_test"]):
        subprocess.run(flags + [str(ROOT / "tests" / (name + ".cpp")), *objects,
                               "-pthread", "-o", str(args.out / name)], check=True)
    if not args.reference:
        subprocess.run(flags + [str(ROOT / "upstream/stockless/tests/search_policy_test.cpp"),
                               "-o", str(args.out / "stockless_policy_regression")], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
