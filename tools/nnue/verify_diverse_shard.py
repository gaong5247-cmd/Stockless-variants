#!/usr/bin/env python3
"""Strict streaming audit of independently generated royal-perspective data.

Validate SHA-256 of source JSONL, per-game rows and every full input
fingerprint before allowing upload to the million-position campaign.
Only standard-library memory proportional to one 1024-game shard.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

GENERATOR="svnnue3-native-selfplay-2"

def canonical(obj):
    return json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def verify(root,variant,shard,profile,min_games):
    manifest_file=root/"manifest.json"
    m=json.loads(manifest_file.read_text(encoding="utf-8"))
    if (m.get("schema")!=3 or m.get("generator")!=GENERATOR
        or m.get("variant")!=variant or m.get("profile")!=profile
        or m.get("shard_id")!=shard or m.get("games",0)<min_games):
        raise ValueError("Wrong generated data identity, variant, shard, profile or game count")
    d=m["descriptor"]
    if (d.get("format_version")!=3 or d.get("feature_version")!=3
        or d.get("variant")!=variant or
        d.get("rules_sha256") is None):
        raise ValueError("Wrong native model descriptor")
    files=("games.jsonl","positions.jsonl","fingerprints.sha256.bin")
    for name in files:
        file=root/name
        expected=m["files"][name]
        if file.stat().st_size!=expected["bytes"] or sha256(file)!=expected["sha256"]:
            raise ValueError(f"Corrupted or incomplete source: {name}")
    if (root/"fingerprints.sha256.bin").stat().st_size!=32*m["positions"]:
        raise ValueError("Expected exactly one full SHA-256 fingerprint per labeled position")

    games={}
    row_expect={}
    game_count=0
    completed=0
    censored=0
    opened=Counter()
    depths=Counter()
    moves=Counter()
    with (root/"games.jsonl").open(encoding="utf-8") as source:
        for line in source:
            g=json.loads(line)
            gid=g.get("game_id")
            if (g.get("schema")!=3 or g.get("variant")!=variant
                or not isinstance(gid,str) or len(gid)!=64 or gid in games
                or not isinstance(g.get("moves"),list)
                or len(g["moves"])!=g.get("plies")
                or not isinstance(g.get("sampling"),dict)
                or g["sampling"].get("profile")!=profile
                or type(g.get("rows")) is not int or g["rows"]<0):
                raise ValueError("Malformed, repeated or incompatible self-play game")
            if g.get("truncated") is True:
                if g.get("result_white") is not None:
                    raise ValueError("Truncated game was falsely labeled a draw")
                censored+=1
            elif g.get("truncated") is False:
                if g.get("result_white") not in (-1,0,1):
                    raise ValueError("Invalid terminal game result")
                completed+=1
            else:
                raise ValueError("Missing termination flag")
            game_count+=1
            games[gid]=g
            row_expect[gid]=g["rows"]
            opened[g["sampling"]["opening_plies"]]+=1
            depths[g["sampling"]["teacher_depth"]]+=1
            moves.update(g.get("move_source_counts",{}))
    if (game_count!=m["games"] or completed!=m["completed_games"]
        or censored!=m["censored_games"]):
        raise ValueError("Game count or native game outcomes mismatch")

    rows=0
    input_set=set()
    with (root/"positions.jsonl").open(encoding="utf-8") as source, \
         (root/"fingerprints.sha256.bin").open("rb") as fingerprint_source:
        for line in source:
            r=json.loads(line)
            gid=r.get("game_id")
            if (r.get("schema")!=3 or gid not in games
                or type(r.get("side_to_move")) is not int
                or r["side_to_move"] not in (0,1)
                or type(r.get("ply")) is not int
                or r.get("trajectory_profile")!=profile
                or r.get("teacher",{}).get("bound") is not None
                or not r.get("fen")
                or r.get("played_move") not in games[gid]["moves"]
                or r.get("played_move_source") not in ("teacher_best","multipv_alternative")):
                raise ValueError("Invalid labeled state, teacher or move provenance")
            score=r.get("teacher_white_cp")
            stm=r.get("teacher_stm_cp")
            if (type(score) not in (float,int) or type(stm) not in (float,int)
                or not -10000<=score<=10000
                or stm!=score*(1 if r["side_to_move"]==0 else -1)):
                raise ValueError("Teacher centipawn label perspective disagreement")
            p=r["ply"]
            if (p<0 or p>=len(games[gid]["moves"])
                or games[gid]["moves"][p]!=r["played_move"]):
                raise ValueError("Label and actual game history misaligned")
            for name in ("features_white","features_black"):
                feature=r.get(name)
                if not isinstance(feature,list) or not feature:
                    raise ValueError("Empty input perspective")
                previous=-1
                total=0
                for pair in feature:
                    if (not isinstance(pair,list) or len(pair)!=2
                        or type(pair[0]) is not int or type(pair[1]) is not int
                        or not previous<pair[0]<d["input_dim"]
                        or not 0<pair[1]<=512):
                        raise ValueError("Corrupted native sparse input pairs")
                    previous=pair[0]
                    total+=pair[1]
                if total>512:raise ValueError("Native royal feature mass exceeds bound")
            fp=hashlib.sha256(canonical([
                r["side_to_move"],r["features_white"],r["features_black"]
            ]).encode()).digest()
            expected_fp=fingerprint_source.read(32)
            if fp!=expected_fp:
                raise ValueError("SHA-256 inputs differ from saved fingerprint sequence")
            input_set.add(fp)
            row_expect[gid]-=1
            rows+=1
        if fingerprint_source.read(1):
            raise ValueError("Extra source input fingerprint after final row")
    if rows!=m["positions"] or any(count!=0 for count in row_expect.values()):
        raise ValueError("Position totals do not agree with game manifests")
    report={"schema":3,"audit_version":"svnnue3-diverse-audit-v1",
        "variant":variant,"profile":profile,"shard":shard,
        "manifest_sha256":sha256(manifest_file),
        "source_engine_sha256":m["native_engine_sha256"],
        "rules_sha256":d["rules_sha256"],"descriptor":d,
        "games":game_count,"positions":rows,
        "distinct_fingerprints_within_shard":len(input_set),
        "completed_games":completed,"censored_games":censored,
        "opening_length_distribution":dict(sorted(opened.items())),
        "teacher_depth_distribution":dict(sorted(depths.items())),
        "move_source_counts":dict(sorted(moves.items())),
        "files":m["files"],"fingerprint_policy":m["fingerprint_policy"],
        "verified":True}
    (root/"audit.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True,type=Path)
    p.add_argument("--variant",required=True)
    p.add_argument("--shard",required=True,type=int)
    p.add_argument("--profile",required=True,choices=("balanced","multipv","deep","exploration","pilot"))
    p.add_argument("--min-games",type=int,default=6)
    a=p.parse_args()
    try:
        print(json.dumps(verify(a.data,a.variant,a.shard,a.profile,a.min_games),
                         sort_keys=True),flush=True)
    except (OSError,ValueError,KeyError,TypeError,IndexError,json.JSONDecodeError) as exc:
        print(f"SHARD INTEGRITY FAILURE: {exc}",file=sys.stderr)
        raise SystemExit(1)
if __name__=="__main__":main()
