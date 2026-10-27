#!/usr/bin/env python3
"""Disk-backed per-variant cardinality and profile-diversity acceptance gate.

Consumes audit-only artifacts (SHA-256 fingerprints and independently checked
source manifests). Every variant must have >=1,000,000 unique model inputs.
Neither target game count nor unverified data is ever reported as success.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

PROFILES=("balanced","multipv","deep","exploration")

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def verify(root,variant,shards,min_unique,min_profiles=4,expected_games=1024,original_root=None):
    if shards<4 or shards%len(PROFILES)!=0:
        raise ValueError('Shard count must be a positive multiple of all four profiles')
    root=Path(root)
    original_root=Path(original_root) if original_root is not None else None
    directory=[folder for folder in root.iterdir()
               if folder.is_dir() and (folder/"audit.json").is_file()]
    if len(directory)!=shards:
        raise ValueError(f"{variant}: expected exactly {shards} certified shards, found {len(directory)}")
    audits={}
    descriptor=None
    teacher=None
    rule=None
    seen_seeds=set()
    seen_profiles=Counter()
    depths=Counter()
    openings=Counter()
    source_counts=Counter()
    total_games=0
    total_positions=0
    finished=0
    censored=0
    for folder in directory:
        audit=json.loads((folder/"audit.json").read_text())
        i=audit.get("shard")
        if (type(i) is not int or not 0<=i<shards or i in audits
            or audit.get("schema")!=3
            or audit.get("audit_version")!="svnnue3-diverse-audit-v1"
            or audit.get("verified") is not True or audit.get("variant")!=variant
            or audit.get("profile") not in PROFILES):
            raise ValueError("Invalid variant/shard/provenance in certified audit")
        seed=audit.get("seed")
        # The original campaign's audit-v1 JSON omitted the seed field,
        # although its already-SHA256-bound native manifest recorded it.
        # Resolve that omission by verifying the ORIGINAL unexpired source
        # artifact, not by inventing/guessing a seed from the shard number.
        if original_root is not None:
            original=original_root/f"v3-diverse-data-{variant}-{i}"
            manifest_file=original/"manifest.json"
            if not manifest_file.is_file():
                raise ValueError(f"Missing original native manifest: {manifest_file}")
            if digest(manifest_file)!=audit.get("manifest_sha256"):
                raise ValueError(f"{variant}/{i}: original signed-manifest SHA256 drift")
            manifest=json.loads(manifest_file.read_text(encoding="utf-8"))
            if (manifest.get("schema")!=3
                or manifest.get("variant")!=variant
                or manifest.get("shard_id")!=i
                or manifest.get("profile")!=audit["profile"]
                or manifest.get("games")!=audit["games"]
                or manifest.get("positions")!=audit["positions"]
                or manifest.get("descriptor")!=audit["descriptor"]
                or manifest.get("native_engine_sha256")!=audit["source_engine_sha256"]):
                raise ValueError(f"{variant}/{i}: original native shard metadata disagree")
            recorded_seed=manifest.get("seed")
            if type(recorded_seed) is not int or (seed is not None and seed!=recorded_seed):
                raise ValueError(f"{variant}/{i}: recorded shard seed mismatch")
            seed=recorded_seed
            for name in ("positions.jsonl","games.jsonl","fingerprints.sha256.bin"):
                source_file=original/name
                audit_info=audit["files"][name]
                native_info=manifest["files"][name]
                if (audit_info!=native_info
                    or source_file.stat().st_size!=audit_info["bytes"]
                    or digest(source_file)!=audit_info["sha256"]):
                    raise ValueError(f"{variant}/{i}: source JSONL or fingerprint checksum mismatch: {name}")
        if type(seed) is not int or seed in seen_seeds:
            raise ValueError(f"{variant}: duplicate or missing independently verified shard seed")
        seen_seeds.add(seed)
        if (audit.get("games")!=expected_games or audit.get("positions",0)<expected_games
            or audit.get("completed_games",0)+audit.get("censored_games",0)!=audit.get("games")):
            raise ValueError("Missing game rows or manufactured censored outcomes")
        if audit.get("fingerprint_policy")!="sha256(canonical([side,features_white,features_black]))":
            raise ValueError("Wrong unique input semantics")
        if descriptor is None:
            descriptor=audit["descriptor"]
            rule=audit["rules_sha256"]
            teacher=audit["source_engine_sha256"]
        if (audit["descriptor"]!=descriptor or audit["rules_sha256"]!=rule
            or audit["source_engine_sha256"]!=teacher):
            raise ValueError("Mixed teacher binary, royal features, or native rules")
        if audit["profile"]!=PROFILES[i//(shards//len(PROFILES))]:
            raise ValueError("Unexpected shard-profile allocation")
        fingerprints=folder/"fingerprints.sha256.bin"
        expected=audit["files"]["fingerprints.sha256.bin"]
        if (not fingerprints.is_file() or fingerprints.stat().st_size!=expected["bytes"]
            or expected["bytes"]!=32*audit["positions"]
            or digest(fingerprints)!=expected["sha256"]):
            raise ValueError(f"Corrupt fingerprint archive {folder}")
        audits[i]={"profile":audit["profile"],"seed":seed,
                   "games":audit["games"],"positions":audit["positions"],
                   "distinct_within_shard":audit["distinct_fingerprints_within_shard"],
                   "manifest_sha256":audit["manifest_sha256"],
                   "fingerprints_sha256":expected["sha256"]}
        total_games+=audit["games"]
        total_positions+=audit["positions"]
        finished+=audit["completed_games"]
        censored+=audit["censored_games"]
        seen_profiles[audit["profile"]]+=audit["games"]
        depths.update({int(k):v for k,v in audit["teacher_depth_distribution"].items()})
        openings.update({int(k):v for k,v in audit["opening_length_distribution"].items()})
        source_counts.update(audit["move_source_counts"])
    if set(audits)!=set(range(shards)):
        raise ValueError("Shard IDs are missing or repeated")
    if len(seen_profiles)<min_profiles or len(depths)<2 or len(openings)<5:
        raise ValueError("Dataset lacked distinct profiles, requested teacher depths, or opening ranges")
    if source_counts["teacher_best"]<=0 or source_counts["opening_random"]<=0:
        raise ValueError("Empty classical teacher or randomized opening source")
    with tempfile.TemporaryDirectory(prefix="svnnue3-audit-") as temp:
        path=Path(temp)/"unique.sqlite3"
        con=sqlite3.connect(path)
        try:
            con.executescript("""PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
                PRAGMA cache_size=-2048; PRAGMA temp_store=FILE;
                CREATE TABLE fingerprints(sha BLOB PRIMARY KEY) WITHOUT ROWID;""")
            for shard in range(shards):
                folder=next(x for x in directory if json.loads((x/"audit.json").read_text())["shard"]==shard)
                with (folder/"fingerprints.sha256.bin").open("rb") as source:
                    while True:
                        block=source.read(32*8192)
                        if not block:break
                        if len(block)%32:
                            raise ValueError("Truncated SHA-256 fingerprint")
                        con.executemany("INSERT OR IGNORE INTO fingerprints(sha) VALUES (?)",
                                        ((sqlite3.Binary(block[i:i+32]),)
                                         for i in range(0,len(block),32)))
                con.commit()
            unique=con.execute("SELECT COUNT(*) FROM fingerprints").fetchone()[0]
        finally:
            con.close()
    result={"schema":3,"variant":variant,"campaign":"svnnue3-diverse-20261027",
            "native_teacher_sha256":teacher,"native_rules_sha256":rule,
            "descriptor":descriptor,"shards":shards,"games":total_games,
            "generated_positions":total_positions,"unique_model_inputs_sha256":unique,
            "duplicate_positions":total_positions-unique,
            "uniqueness_ratio":round(unique/total_positions,6),
            "profiles_games":dict(seen_profiles),
            "teacher_depth_games":dict(sorted(depths.items())),
            "opening_lengths_games":dict(sorted(openings.items())),
            "trajectory_sources":dict(source_counts),
            "completed_games":finished,"censored_games":censored,
            "source_shards":[{"id":i,**audits[i]} for i in range(shards)],
            "unique_target":min_unique,"unique_target_met":unique>=min_unique,
            "fingerprint_algorithm":"full SHA-256 over native perspective features + side"}
    if unique<min_unique:
        raise ValueError(f"{variant}: only {unique:,} distinct model inputs, target {min_unique:,}. Generated {total_positions:,}.")
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--variant",required=True,choices=("chess","shogi","janggi","xiangqi","makruk"))
    p.add_argument("--shards",type=int,default=12)
    p.add_argument("--min-unique",type=int,default=1000000)
    p.add_argument("--expected-games",type=int,default=1024)
    p.add_argument("--original-root",type=Path,
                   help="Original whole source artifact folders, required for recovering audit-v1 missing seeds")
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    try:
        result=verify(args.input_root,args.variant,args.shards,args.min_unique,
                      expected_games=args.expected_games,original_root=args.original_root)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        print(json.dumps({k:result[k] for k in
            ("variant","games","generated_positions","unique_model_inputs_sha256",
             "duplicate_positions","unique_target_met","profiles_games",
             "teacher_depth_games","trajectory_sources")}),flush=True)
    except (ValueError,OSError,KeyError,TypeError,IndexError,json.JSONDecodeError) as exc:
        print(f"MILLION-POSITION VERIFICATION FAILED: {exc}",file=sys.stderr)
        raise SystemExit(1)
if __name__=="__main__":main()
