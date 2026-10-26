#!/usr/bin/env python3
"""Package a binary with its exact generated source, licenses and capability evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build/engine")
    parser.add_argument("--platform", default=sys.platform)
    parser.add_argument("--arch", default="x86-64")
    parser.add_argument("--report", type=Path, action="append", default=[])
    parser.add_argument("--catalog", type=Path, default=ROOT / "docs/variants.json")
    args = parser.parse_args()
    engine = args.engine.resolve()
    build = args.build_dir.resolve()
    if not engine.is_file():
        parser.error(f"Missing executable: {engine}")
    manifest = json.loads((build / "build-manifest.json").read_text())
    for name, expected in manifest["source_files"].items():
        if sha256(build / name) != expected:
            raise SystemExit(f"Generated source changed after materialization: {name}; rebuild before packaging")
    if args.catalog.is_file():
        catalog = json.loads(args.catalog.read_text())
    else:
        raise SystemExit("Generate the source/runtime variant catalog before packaging")
    engine_hash = sha256(engine)
    runtime = catalog.get("runtime")
    if runtime and runtime.get("engine_sha256") != engine_hash:
        raise SystemExit("Runtime catalog belongs to a different binary; regenerate it")
    for report_path in args.report:
        report = json.loads(report_path.read_text())
        if report.get("failed") != 0 or report.get("passed", 0) <= 0:
            raise SystemExit(f"Validation report failed or has no passing cases: {report_path}")
        if report.get("engine_sha256") != engine_hash:
            raise SystemExit(f"Validation report belongs to a different binary: {report_path}")
    for value in (manifest["version"], args.platform, args.arch):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
            raise SystemExit(f"Unsafe package label: {value!r}")

    label = f"stockless-variants-{manifest['version']}-{args.platform}-{args.arch}"
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    stage = dist / label
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    shutil.copy2(engine, stage / engine.name)
    for name in ("LICENSE", "NOTICE.md", "README.md"):
        shutil.copy2(ROOT / name, stage / name)
    shutil.copytree(ROOT / "LICENSES", stage / "LICENSES")
    shutil.copytree(ROOT / "config", stage / "config")
    shutil.copy2(args.catalog, stage / "variants.json")
    shutil.copytree(ROOT / "docs", stage / "docs")
    shutil.copy2(build / "AUTHORS", stage / "AUTHORS.FAIRY-STOCKFISH")
    shutil.copy2(build / "Copying.txt", stage / "Copying.FAIRY-STOCKFISH.txt")
    (stage / "validation").mkdir()
    for report in args.report:
        shutil.copy2(report, stage / "validation" / report.name)

    # Full corresponding generated source is supplied alongside each binary;
    # network retrieval/submodule availability is not required to inspect it.
    compiler_args = " COMP=mingw" if args.platform == "windows" else ""
    compiler_setup = ""
    if args.platform == "android-arm64":
        compiler_args = " COMP=ndk CXX=aarch64-linux-android26-clang++"
        compiler_setup = (
            "Android: install NDK 27.2.12479018, then put its\n"
            "toolchains/llvm/prebuilt/linux-x86_64/bin directory on PATH.\n"
            "This recipe targets ARM64 Android API 26 using the static C++ runtime.\n"
        )
    source_zip = stage / "corresponding-source.zip"
    with zipfile.ZipFile(source_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in manifest["source_files"]:
            archive.write(build / name, name)
        for name in ("AUTHORS", "Copying.txt", "LICENSE.STOCKLESS-VARIANTS", "NOTICE.STOCKLESS-VARIANTS.md", "build-manifest.json"):
            archive.write(build / name, name)
        archive.writestr("BUILD.txt", f"Stockless Variants {manifest['version']}\nSource commit: {manifest['commit']}\n"
                          + compiler_setup
                          + f"make -C src -j2 build ARCH={args.arch}{compiler_args} largeboards=yes all=yes nnue=no EXE={engine.name}\n"
                          "No neural network is embedded or required. See the supplied licenses and NOTICE.\n")

    packaged = {"version": manifest["version"], "commit": manifest["commit"],
                "working_tree_dirty": manifest.get("working_tree_dirty", True),
                "status": "experimental-development-build", "platform": args.platform,
                "architecture": args.arch, "engine_sha256": sha256(engine),
                "evaluation": "classical", "nnue": manifest.get("nnue"),
                "upstream": manifest["upstream"],
                "validation_files": [p.name for p in args.report],
                "runtime_tested_in_this_package": bool(args.report),
                "files": {p.relative_to(stage).as_posix(): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}}
    (stage / "manifest.json").write_text(json.dumps(packaged, indent=2) + "\n")
    (stage / "SOURCE.md").write_text(
        "# Corresponding source\n\nThe complete materialized C++ source for this binary is in `corresponding-source.zip`.\n\n"
        f"Overlay repository: https://github.com/gaong5247-cmd/Stockless-variants/tree/{manifest['commit']}\n\n"
        "All components retain their original notices. This is an experimental build; the variant catalog separates available rules from validation and special-architecture work.\n")
    archive_path = Path(shutil.make_archive(str(stage), "zip", root_dir=stage))
    # Reopen and verify every manifest entry, not only that the ZIP exists.
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise SystemExit("Package ZIP integrity check failed")
        for name, expected in packaged["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise SystemExit(f"Package hash mismatch: {name}")
    archive_path.with_suffix(".zip.sha256").write_text(f"{sha256(archive_path)}  {archive_path.name}\n")
    print(archive_path)


if __name__ == "__main__":
    main()
