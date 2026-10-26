#!/usr/bin/env python3
"""Merge independently generated SVNNUE-v1 shards with strict provenance checks.

Only identical teacher executables, rules, feature descriptors and generator
settings may be combined. Truncated results stay null, never become draws.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def merge(shards: list[Path], output: Path, seed: int) -> dict:
    if len(shards) < 2:
        raise ValueError("At least two independent generation shards required")
    if len(set(path.resolve() for path in shards)) != len(shards):
        raise ValueError("Duplicate shard paths")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty")
    common = None
    inspected = []
    seen_ids = set()
    for shard in shards:
        manifest_path = shard / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != 1 or manifest.get("generator_version") != "svnnue-selfplay-v1":
            raise ValueError(f"Unsupported generation manifest: {shard}")
        if manifest.get("source_working_tree_dirty") is True:
            raise ValueError("Refusing to mix data from dirty source builds")
        identity = {key: manifest.get(key) for key in (
            "variant", "descriptor", "engine_sha256", "rules_sha256",
            "teacher", "source_commit", "source_working_tree_dirty",
            "build_source_files_sha256", "build_manifest_sha256",
        )}
        identity["limits_except_games"] = {
            key: val for key, val in manifest.get("limits", {}).items() if key != "games"
        }
        if common is None:
            common = identity
        elif canonical(identity) != canonical(common):
            raise ValueError(f"Teacher, feature, rules, source, or limits mismatch: {shard}")
        files = {}
        for filename in ("games.jsonl", "positions.jsonl"):
            source = shard / filename
            expected = manifest.get("files", {}).get(filename, {})
            if (not source.is_file() or checksum(source) != expected.get("sha256")
                    or source.stat().st_size != expected.get("bytes")):
                raise ValueError(f"Shard checksum/length mismatch: {source}")
            files[filename] = source
        game_ids = set()
        game_count = complete = truncated = 0
        with files["games.jsonl"].open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    raise ValueError("Blank game line")
                row = json.loads(line)
                gid = row.get("game_id")
                if not isinstance(gid, str) or len(gid) != 64 or gid in seen_ids:
                    raise ValueError("Invalid or duplicate game identity across shards")
                seen_ids.add(gid)
                game_ids.add(gid)
                game_count += 1
                complete += row.get("truncated") is False
                truncated += row.get("truncated") is True
        positions = 0
        with files["positions.jsonl"].open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    raise ValueError("Blank position line")
                row = json.loads(line)
                if row.get("game_id") not in game_ids:
                    raise ValueError("Position refers to a game outside its shard")
                positions += 1
        if (manifest.get("game_count") != game_count or manifest.get("rows") != positions
                or manifest.get("completed_games") != complete
                or manifest.get("truncated_games") != truncated
                or manifest.get("limits", {}).get("games") != game_count):
            raise ValueError("Shard record counts differ from manifest")
        inspected.append((shard, manifest, files))
    assert common is not None
    output.mkdir(parents=True, exist_ok=True)
    combined = copy.deepcopy(inspected[0][1])
    combined["generator_version"] = "svnnue-merged-shards-v1"
    combined["seed"] = seed
    for name in ("game_count", "rows", "completed_games", "truncated_games", "skipped_labels"):
        combined[name] = sum(manifest[name] for _, manifest, _ in inspected)
    combined["limits"]["games"] = combined["game_count"]
    combined["shard_provenance"] = [
        {
            "dataset_manifest_sha256": checksum(shard / "manifest.json"),
            "seed": m["seed"], "games": m["game_count"], "positions": m["rows"],
            "games_sha256": m["files"]["games.jsonl"]["sha256"],
            "positions_sha256": m["files"]["positions.jsonl"]["sha256"],
        }
        for shard, m, _ in inspected
    ]
    for filename in ("games.jsonl", "positions.jsonl"):
        target = output / filename
        with target.open("xb") as sink:
            for _, _, files in inspected:
                with files[filename].open("rb") as source:
                    shutil.copyfileobj(source, sink, length=1024 * 1024)
        combined["files"][filename] = {
            "sha256": checksum(target), "bytes": target.stat().st_size
        }
    (output / "manifest.json").write_text(
        json.dumps(combined, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return combined


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=("chess", "shogi", "janggi", "xiangqi", "makruk"), required=True)
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--shards", type=int, default=2)
    args = p.parse_args()
    if not 0 <= args.seed < 2**32 or not 2 <= args.shards <= 16:
        p.error("seed must be uint32 and shards 2..16")
    sources = sorted(path.parent for path in args.input_root.glob("*/manifest.json"))
    if len(sources) != args.shards:
        p.error(f"expected exactly {args.shards} shard manifests, got {len(sources)}")
    try:
        result = merge(sources, args.output, args.seed)
        if result["variant"] != args.variant:
            raise ValueError("Merged variant differs from requested variant")
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        print(f"Merge failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"variant": args.variant, "games": result["game_count"],
                      "positions": result["rows"], "complete": result["completed_games"],
                      "truncated": result["truncated_games"], "shards": len(sources)},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
