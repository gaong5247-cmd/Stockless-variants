#!/usr/bin/env python3
"""Re-encode audited native v2 teacher rows with genuine SVNNUE-v3 features.

No new teacher search or fake new games. Preserves original depth-4 teacher
labels and game results verbatim, proves source hashes and replays full native
histories in the exact new C++ position engine.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from generate import Engine, validate_state, sha256, canonical, build_metadata, declared_source_identity


def reencode(source: Path, target: Path, engine_path: Path, timeout: float):
    manifest_path=source/"manifest.json"
    original=json.loads(manifest_path.read_text(encoding="utf-8"))
    if (original.get("schema")!=1 or
            original.get("descriptor",{}).get("feature_version")!=2 or
            original.get("descriptor",{}).get("format_version")!=2):
        raise ValueError("Reencoder requires authentic SVNNUE-v2 source datasets")
    for filename in ("positions.jsonl","games.jsonl"):
        path=source/filename
        expected=original["files"][filename]
        if not path.exists() or path.stat().st_size!=expected["bytes"] or sha256(path)!=expected["sha256"]:
            raise ValueError("Original teacher source checksum mismatch: "+filename)
    if original.get("source_working_tree_dirty") is True:
        raise ValueError("Refusing source from a dirty teacher")
    engine_hash=sha256(engine_path)
    build,_=build_metadata(engine_path)
    variant=original["variant"]
    teacher=Engine(engine_path,timeout)
    teacher.feature_version=3
    started=time.monotonic()
    games={}
    newids={}
    descriptor=None
    try:
        teacher.option("UCI_Variant",variant)
        teacher.option("Threads",1)
        teacher.option("EvalBackend","Classical")
        for line in (source/"games.jsonl").open(encoding="utf-8"):
            game=json.loads(line)
            old_id=game["game_id"]
            if old_id in games:
                raise ValueError("Duplicate original game identity")
            if game["variant"]!=variant or game["teacher_engine_sha256"]!=original["engine_sha256"]:
                raise ValueError("Teacher source game provenance mismatch")
            games[old_id]=game
        if len(games)!=original["game_count"]:
            raise ValueError("Source game count mismatch")
        target.mkdir(parents=True,exist_ok=True)
        if any(target.iterdir()):
            raise ValueError("Output directory must be empty")
        out_pos=target/"positions.jsonl"
        out_game=target/"games.jsonl"
        counts={key:0 for key in games}
        changed=0
        with (source/"positions.jsonl").open(encoding="utf-8") as inp, \
                out_pos.open("x",encoding="utf-8") as writer:
            for line in inp:
                row=json.loads(line)
                old_id=row["game_id"]
                game=games.get(old_id)
                if game is None:
                    raise ValueError("Position belongs to foreign source game")
                moves=game["moves"][:row["ply"]]
                teacher.position(moves)
                native=teacher.features()
                descriptor=validate_state(native,variant,descriptor)
                if descriptor["feature_version"]!=3 or descriptor["format_version"]!=3:
                    raise ValueError("Native engine did not export true v3 features")
                if native["fen"]!=row["fen"] or native["side_to_move"]!=row["side_to_move"]:
                    raise ValueError("Replayed legal state changed compared to original v2 teacher position")
                game_id=newids.setdefault(old_id,hashlib.sha256(canonical({
                    "source_game_id":old_id,"variant":variant,
                    "v3_native_rules_sha256":descriptor["rules_sha256"],
                    "native_version":3
                }).encode()).hexdigest())
                fingerprint=hashlib.sha256(canonical({
                    "rules_sha256":descriptor["rules_sha256"],
                    "feature_version":3,"side_to_move":native["side_to_move"],
                    "features":native["features"]
                }).encode()).hexdigest()
                rewrite=dict(row)
                rewrite["game_id"]=game_id
                rewrite["source_game_id"]=old_id
                rewrite["source_rules_sha256"]=row["rules_sha256"]
                rewrite["source_feature_sha256"]=row.get("feature_sha256")
                rewrite["rules_sha256"]=descriptor["rules_sha256"]
                rewrite["features"]=native["features"]
                rewrite["feature_sha256"]=fingerprint
                rewrite["native_feature_version"]=3
                writer.write(canonical(rewrite)+"\n")
                counts[old_id]+=1
                changed+=1
                if changed%10000==0:
                    print(json.dumps({"variant":variant,"reencoded_rows":changed,"elapsed_s":round(time.monotonic()-started,2)}),flush=True)
        if changed!=original["rows"]:
            raise ValueError("Original labeled-position count changed during rematerialization")
        with out_game.open("x",encoding="utf-8") as writer:
            for old_id,game in games.items():
                if counts[old_id]!=game["rows"]:
                    raise ValueError("Game/position counts differ from original teacher")
                copy=dict(game)
                copy["source_game_id"]=old_id
                copy["game_id"]=newids.get(old_id,hashlib.sha256(canonical({
                    "source_game_id":old_id,"variant":variant,
                    "v3_native_rules_sha256":descriptor["rules_sha256"],"native_version":3
                }).encode()).hexdigest())
                copy["source_rules_sha256"]=game["rules_sha256"]
                copy["rules_sha256"]=descriptor["rules_sha256"]
                writer.write(canonical(copy)+"\n")
        newmanifest=dict(original)
        newmanifest.update({
            "generator_version":"svnnue-native-reencode-v3",
            "descriptor":descriptor,"rules_sha256":descriptor["rules_sha256"],
            "source_data_manifest_sha256":sha256(manifest_path),
            "source_data_feature_version":2,
            "feature_exporter_engine_sha256":engine_hash,
            "feature_exporter_engine_id":teacher.engine_id,
            "feature_exporter_source_commit":build.get("commit") if build else None,
            "feature_exporter_source_dirty":build.get("working_tree_dirty") if build else None,
            "source_teacher_rules_sha256":original["rules_sha256"],
            "reencoded_positions":changed,
            "native_feature_version":3,
            "relabeling":False,
            "original_teacher_scores_preserved":True,
            "elapsed_seconds":round(time.monotonic()-started,2),
            "files": {name:{"sha256":sha256(target/name),"bytes":(target/name).stat().st_size}
                      for name in ("positions.jsonl","games.jsonl")}
        })
        (target/"manifest.json").write_text(json.dumps(newmanifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
        return {"variant":variant,"games":len(games),"positions":changed,
                "old_rules":original["rules_sha256"],"new_rules":descriptor["rules_sha256"],
                "teacher_sha256":original["engine_sha256"],"exporter_sha256":engine_hash}
    finally:
        teacher.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--engine",required=True,type=Path)
    p.add_argument("--timeout",type=float,default=90)
    args=p.parse_args()
    if not args.engine.is_file() or not args.source.is_dir():
        p.error("Missing native engine or genuine v2 dataset")
    try:
        print(json.dumps(reencode(args.source,args.output,args.engine.resolve(),args.timeout)),flush=True)
    except (OSError,ValueError,RuntimeError,KeyError,TypeError,AssertionError) as exc:
        print("NATIVE REENCODE FAILED:",str(exc),file=sys.stderr)
        raise


if __name__=="__main__":
    main()
