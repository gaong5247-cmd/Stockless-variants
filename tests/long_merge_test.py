#!/usr/bin/env python3
"""Small standard-library regression for long NNUE shard merge."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location(
    "merge_long", Path(__file__).resolve().parents[1] / "tools/long_training/merge.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_shard(folder, index, engine="a" * 64):
    folder.mkdir(parents=True, exist_ok=True)
    gid = hashlib.sha256(f"game-{index}".encode()).hexdigest()
    files = {}
    for name, record in (
        ("games.jsonl", {"game_id": gid, "truncated": True}),
        ("positions.jsonl", {"game_id": gid, "features": [[0, 1]]}),
    ):
        path = folder / name
        path.write_text(json.dumps(record) + "\n")
        files[name] = {"sha256": module.checksum(path), "bytes": path.stat().st_size}
    manifest = {
        "schema": 1, "generator_version": "svnnue-selfplay-v1",
        "variant": "chess", "descriptor": {"variant": "chess"},
        "engine_sha256": engine, "rules_sha256": "0"*64,
        "teacher": {"engine_sha256": engine},
        "source_commit": "a"*40, "source_working_tree_dirty": False,
        "build_source_files_sha256": "b"*64, "build_manifest_sha256": "c"*64,
        "limits": {"games": 1, "depth": 4, "max_plies": 136},
        "seed": 1000 + index, "game_count": 1, "rows": 1,
        "completed_games": 0, "truncated_games": 1, "skipped_labels": 0,
        "files": files,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest))


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a, b = root / "s0", root / "s1"
        write_shard(a, 0)
        write_shard(b, 1)
        combined = module.merge([a, b], root / "merged", 123)
        assert combined["game_count"] == combined["rows"] == 2
        assert len(combined["shard_provenance"]) == 2
        assert module.checksum(root / "merged/positions.jsonl") == combined["files"]["positions.jsonl"]["sha256"]
        try:
            module.merge([a, a], root / "duplicate-path", 123)
            raise AssertionError("Duplicate shard accepted")
        except ValueError:
            pass
        write_shard(b, 0)
        try:
            module.merge([a, b], root / "duplicate-game", 123)
            raise AssertionError("Repeated game accepted")
        except ValueError:
            pass
        write_shard(b, 1, "d" * 64)
        try:
            module.merge([a, b], root / "different-teacher", 123)
            raise AssertionError("Different teacher accepted")
        except ValueError:
            pass
        write_shard(b, 1)
        with (b / "positions.jsonl").open("a") as f:
            f.write("{}\n")
        try:
            module.merge([a, b], root / "corrupt", 123)
            raise AssertionError("Corrupted shard accepted")
        except ValueError:
            pass
    print("PASS shard merge: count, provenance, duplicate IDs and corruption")


if __name__ == "__main__":
    main()
