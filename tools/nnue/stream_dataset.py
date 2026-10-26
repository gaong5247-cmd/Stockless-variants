#!/usr/bin/env python3
"""Disk-backed, bounded-memory index for million-row SVNNUE-v1 distillation.

No dense (rows x input_dim) array is materialized; integrity is checked before
insertion, whole-game partitions and input deduplication are immutable.
Copyright (C) 2026 Stockless Variants contributors.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import struct

FORMAT = "stockless-stream-index-v1"
BUCKETS = 64
PACK = struct.Struct("<HH")
PARTITIONS = ("train", "validation", "test")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_info(paths, seed):
    if not paths:
        raise ValueError("At least one dataset is required")
    items = []
    reference = None
    for path in map(Path, paths):
        mpath = path / "manifest.json"
        m = json.loads(mpath.read_text(encoding="utf-8"))
        d = m.get("descriptor")
        if not isinstance(d, dict) or not d.get("supported") or m.get("schema") != 1:
            raise ValueError("Invalid native dataset manifest")
        if m.get("source_working_tree_dirty") is True:
            raise ValueError("Dirty teacher source is not supported for training")
        width, height, ids = d["width"], d["height"], d["type_ids"]
        if not (1 <= width <= 12 and 1 <= height <= 10 and ids == sorted(set(ids))
                and d["input_dim"] == 2 * len(ids) * (width * height + 1)):
            raise ValueError("Invalid native feature descriptor")
        common = {k: m.get(k) for k in ("variant", "rules_sha256", "engine_sha256", "descriptor")}
        common["teacher"] = m.get("teacher")
        if reference is None:
            reference = common
        elif canonical(common) != canonical(reference):
            raise ValueError("Dataset teacher, native features or rules disagree")
        if m["variant"] not in ("chess", "shogi", "janggi", "xiangqi", "makruk"):
            raise ValueError("Unsupported SVNNUE training variant")
        for filename in ("games.jsonl", "positions.jsonl"):
            record = m.get("files", {}).get(filename, {})
            actual = path / filename
            if actual.stat().st_size != record.get("bytes") or digest(actual) != record.get("sha256"):
                raise ValueError(f"Missing or corrupted file: {actual}")
        items.append({"path": str(path.resolve()), "manifest_sha256": digest(mpath),
                      "manifest": m, "games": path / "games.jsonl",
                      "positions": path / "positions.jsonl"})
    assert reference is not None
    identity = hashlib.sha256(canonical({
        "format": FORMAT, "seed": seed, "manifests": [i["manifest_sha256"] for i in items],
        "reference": reference
    }).encode()).hexdigest()
    return items, reference["descriptor"], reference, identity


def split_for_game(game_id, seed):
    value = hashlib.sha256(f"SV-SPLIT:{seed}:{game_id}".encode()).digest()[0] % 6
    return 1 if value == 0 else (2 if value == 1 else 0)


def encode_features(features, dimension, area):
    if not isinstance(features, list) or not features:
        raise ValueError("Missing sparse NNUE features")
    result = bytearray()
    prev = -1
    mass = 0
    for entry in features:
        if not (isinstance(entry, list) and len(entry) == 2
                and type(entry[0]) is int and type(entry[1]) is int):
            raise ValueError("Bad native feature pair")
        idx, value = entry
        if not (prev < idx < dimension and 1 <= value <= 512
                and (idx % (area+1) == area or value == 1)):
            raise ValueError("Unsorted, duplicate, malformed or out-of-range native feature")
        mass += value
        prev = idx
        result.extend(PACK.pack(idx, value))
    if mass > 512:
        raise ValueError("Native feature mass exceeds bound")
    return bytes(result)


def decode_features(blob):
    if len(blob) % PACK.size:
        raise ValueError("Malformed feature blob")
    return [pair for pair in struct.iter_unpack("<HH", blob)]


def open_db(path):
    db = sqlite3.connect(str(path), timeout=60)
    db.execute("PRAGMA cache_size=-4096")
    db.execute("PRAGMA temp_store=FILE")
    return db


def build_index(paths, output, seed=20261026):
    output = Path(output)
    items, desc, reference, signature = manifest_info(paths, seed)
    if output.exists():
        with open_db(output) as db:
            row = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
            if not row or row[0] != signature:
                raise ValueError("Refusing to reuse an index with different data/seed")
            return json.loads(db.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".building")
    if staging.exists():
        staging.unlink()  # An uncommitted index is never a resume checkpoint.
    db = open_db(staging)
    try:
        db.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE games (
              game_id TEXT PRIMARY KEY, partition INTEGER NOT NULL,
              moves TEXT NOT NULL, variant TEXT NOT NULL,
              source INTEGER NOT NULL, rows_expected INTEGER NOT NULL,
              rows_seen INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE examples (
              id INTEGER PRIMARY KEY,
              input_key TEXT NOT NULL UNIQUE,
              partition INTEGER NOT NULL, bucket INTEGER NOT NULL,
              side INTEGER NOT NULL, label REAL NOT NULL,
              features BLOB NOT NULL,
              game_id TEXT NOT NULL, ply INTEGER NOT NULL, fen TEXT NOT NULL
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        game_counts = [0, 0, 0]
        declared_rows = 0
        for si, item in enumerate(items):
            m = item["manifest"]
            n_games = 0
            with item["games"].open(encoding="utf-8") as f:
                for line in f:
                    game = json.loads(line)
                    gid = game.get("game_id")
                    if not (isinstance(gid, str) and len(gid) == 64 and
                            game.get("variant") == m["variant"] and
                            game.get("teacher_engine_sha256") == m["engine_sha256"] and
                            game.get("rules_sha256") == m["rules_sha256"] and
                            isinstance(game.get("moves"), list) and
                            len(game["moves"]) == game["plies"] and
                            type(game.get("rows")) is int and game["rows"] >= 0):
                        raise ValueError("Game provenance or length invalid")
                    if game.get("truncated") is True:
                        if game.get("result_white") is not None:
                            raise ValueError("Truncated games must have null outcome")
                    elif game.get("truncated") is not False:
                        raise ValueError("Bad game termination flag")
                    part = split_for_game(gid, seed)
                    try:
                        db.execute("INSERT INTO games(game_id,partition,moves,variant,source,rows_expected) VALUES(?,?,?,?,?,?)",
                                   (gid, part, json.dumps(game["moves"]), m["variant"], si, game["rows"]))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError("Duplicate game identity across shards") from exc
                    game_counts[part] += 1
                    declared_rows += game["rows"]
                    n_games += 1
            if n_games != m["game_count"]:
                raise ValueError("Game count mismatch")
        db.commit()
        row_counts = [0, 0, 0]
        duplicates = [0, 0, 0]
        # Partition priority protects validation/test against training leakage.
        for partition in range(3):
            for si, item in enumerate(items):
                m = item["manifest"]
                area = desc["width"] * desc["height"]
                actual_rows = 0
                with item["positions"].open(encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        actual_rows += 1
                        gid = row.get("game_id")
                        game = db.execute("SELECT partition,source,rows_expected FROM games WHERE game_id=?",
                                          (gid,)).fetchone()
                        if game is None or game[1] != si:
                            raise ValueError("Position references a missing or foreign game")
                        if game[0] != partition:
                            continue
                        raw = row.get("teacher")
                        side = row.get("side_to_move")
                        label = row.get("label_cp_white")
                        if (row.get("variant") != m["variant"]
                                or row.get("rules_sha256") != m["rules_sha256"]
                                or row.get("teacher_engine_sha256") != m["engine_sha256"]
                                or type(side) is not int or side not in (0, 1)
                                or not isinstance(raw, dict) or raw.get("unit") not in ("cp", "mate")
                                or raw.get("bound") is not None or type(raw.get("value")) is not int
                                or not isinstance(label, (float, int))
                                or type(row.get("ply")) is not int
                                or row["ply"] < 0 or not isinstance(row.get("fen"), str)
                                or not row["fen"]):
                            raise ValueError("Malformed teacher labeled position")
                        expected = (10000 if raw["value"] > 0 else -10000) if raw["unit"] == "mate" \
                            else max(-10000, min(10000, raw["value"]))
                        expected *= 1 if side == 0 else -1
                        if label != expected:
                            raise ValueError("Teacher label and perspective mismatch")
                        features = row["features"]
                        blob = encode_features(features, desc["input_dim"], area)
                        key = hashlib.sha256(json.dumps([side, features], separators=(",", ":")).encode()).hexdigest()
                        bucket = int(key[:2], 16) % BUCKETS
                        cursor = db.execute(
                            "INSERT OR IGNORE INTO examples(input_key,partition,bucket,side,label,features,game_id,ply,fen)"
                            " VALUES(?,?,?,?,?,?,?,?,?)",
                            (key,partition,bucket,side,float(max(-2000,min(2000,label))),blob,gid,row["ply"],row["fen"]))
                        if cursor.rowcount:
                            row_counts[partition] += 1
                        else:
                            duplicates[partition] += 1
                        db.execute("UPDATE games SET rows_seen=rows_seen+1 WHERE game_id=?", (gid,))
                if partition == 0 and actual_rows != m["rows"]:
                    raise ValueError("Manifest position count mismatch")
            db.commit()
        wrong = db.execute("SELECT game_id FROM games WHERE rows_expected!=rows_seen LIMIT 1").fetchone()
        if wrong:
            raise ValueError("Labeled position count disagrees with game manifest: " + wrong[0])
        if sum(row_counts) + sum(duplicates) != declared_rows:
            raise ValueError("Total input rows mismatch")
        if min(row_counts) < 8:
            raise ValueError("Missing train/validation/test positions")
        db.execute("CREATE INDEX example_batches ON examples(partition,bucket,id)")
        summary = {"format": FORMAT, "signature": signature, "variant": desc["variant"],
                   "descriptor": desc, "teacher": reference["teacher"],
                   "source_manifests": [i["manifest_sha256"] for i in items],
                   "partition_rows": dict(zip(PARTITIONS, row_counts)),
                   "duplicates_removed": dict(zip(PARTITIONS, duplicates)),
                   "game_partitions": dict(zip(PARTITIONS, game_counts)),
                   "rows_read": declared_rows, "rows_indexed": sum(row_counts),
                   "seed": seed, "database": str(output)}
        for name, value in {"signature": signature, "summary": canonical(summary)}.items():
            db.execute("INSERT INTO metadata(key,value) VALUES(?,?)", (name,value))
        db.commit()
    except BaseException:
        db.close()
        if staging.exists():
            staging.unlink()
        raise
    else:
        db.close()
        staging.replace(output)
        return summary


def batches(db, partition, batch_size, epoch_seed=0):
    import numpy as np
    buckets = np.random.default_rng(epoch_seed).permutation(BUCKETS)
    for bucket in buckets:
        last = 0
        while True:
            found = db.execute(
                "SELECT id,features,side,label,game_id,ply,fen FROM examples"
                " WHERE partition=? AND bucket=? AND id>? ORDER BY id LIMIT ?",
                (partition, int(bucket), last, batch_size)).fetchall()
            if not found:
                break
            last = found[-1][0]
            yield found


def dense_batch(batch, dim):
    import numpy as np
    x = np.zeros((len(batch), dim), dtype=np.float32)
    sides = np.empty(len(batch), dtype=np.int64)
    labels = np.empty(len(batch), dtype=np.float32)
    for i, (_, packed, side, label, *_rest) in enumerate(batch):
        for idx, val in decode_features(packed):
            x[i, idx] = val
        sides[i] = side
        labels[i] = label
    return x, sides, labels


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, nargs="+")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20261026)
    args = p.parse_args()
    print(json.dumps(build_index(args.data,args.index,args.seed), sort_keys=True))
