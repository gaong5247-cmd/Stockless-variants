#!/usr/bin/env python3
"""Bounded-RAM merge of independently certified SVNNUE3 native shard files.

The resulting manifest preserves original shard hashes and all 4 teacher
sampling distributions. No label, game outcome or NNUE vector is changed.
The merged JSONL files are usable by dataset_v3.py and train_v3.py.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

from verify_diverse_shard import verify

PROFILES=("balanced","multipv","deep","exploration")

def digest(path):
    h=hashlib.sha256()
    with open(path,"rb") as handle:
        for part in iter(lambda:handle.read(1024*1024),b""):
            h.update(part)
    return h.hexdigest()

def merge(root,output,variant,shards=12,games_per_shard=1024):
    root=Path(root)
    output=Path(output)
    if shards<4 or shards%4!=0:
        raise ValueError("Expected four entire profile groups")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Never overwrite an existing merged NNUE dataset")
    expected={}
    teacher=None
    descriptor=None
    settings=None
    positions=0
    games=0
    complete=0
    unfinished=0
    seen_seeds=set()
    records=[]
    for shard in range(shards):
        candidates=[root/str(shard),root/f"v3-diverse-data-{variant}-{shard}"]
        options=[p for p in candidates if (p/"manifest.json").is_file()]
        if len(options)!=1:
            raise ValueError(f"Cannot find exactly one original source for shard {shard}")
        folder=options[0]
        profile=PROFILES[shard//(shards//4)]
        audit=verify(folder,variant,shard,profile,games_per_shard)
        manifest=json.loads((folder/"manifest.json").read_text())
        if manifest["games"]!=games_per_shard:
            raise ValueError("Unexpected source game count")
        if (teacher is not None and teacher!=manifest["native_engine_sha256"] or
            descriptor is not None and descriptor!=manifest["descriptor"]):
            raise ValueError("Incompatible network features or native teacher across shards")
        teacher=manifest["native_engine_sha256"]
        descriptor=manifest["descriptor"]
        seed=manifest["seed"]
        if seed in seen_seeds:
            raise ValueError("Repeated native self-play seed across shards")
        seen_seeds.add(seed)
        rows=manifest["positions"]
        games+=manifest["games"]
        positions+=rows
        complete+=manifest["completed_games"]
        unfinished+=manifest["censored_games"]
        records.append({"folder":folder,"profile":profile,"shard":shard,
                        "seed":seed,"games":manifest["games"],"positions":rows,
                        "manifest_sha256":digest(folder/"manifest.json"),
                        "fingerprints_sha256":digest(folder/"fingerprints.sha256.bin")})
    if not records or games!=shards*games_per_shard:
        raise ValueError("Missing original native game histories")
    output.mkdir(parents=True,exist_ok=True)
    files={}
    for name in ("games.jsonl","positions.jsonl","fingerprints.sha256.bin"):
        target=output/name
        with target.open("xb") as destination:
            for record in records:
                with (record["folder"]/name).open("rb") as source:
                    shutil.copyfileobj(source,destination,1024*1024)
        files[name]={"sha256":digest(target),"bytes":target.stat().st_size}
    if files["fingerprints.sha256.bin"]["bytes"]!=32*positions:
        raise ValueError("Merged fingerprint count disagrees with native position count")
    merged={
        "schema":3,"generator":"svnnue3-native-selfplay-2",
        "merge_version":"svnnue3-diverse-stream-merge-v1",
        "descriptor":descriptor,"variant":variant,
        "seed":20261027,"native_engine_sha256":teacher,
        "teacher_options":{"EvalBackend":"Classical","teacher_label_scope":"same exact native engine",
                          "depths":[2,3,4,5],"MultiPV_choices":[1,2,3]},
        "games":games,"positions":positions,
        "completed_games":complete,"censored_games":unfinished,
        "split_policy":"whole-game hash based split and duplicate-free retained native perspective states",
        "source_shards":[{k:v for k,v in r.items() if k!="folder"} for r in records],
        "files":files
    }
    (output/"manifest.json").write_text(json.dumps(merged,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    return {"variant":variant,"games":games,"rows":positions,
            "files":files,"source_manifests":len(records),
            "merged_manifest_sha256":digest(output/"manifest.json")}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--variant",required=True,choices=PROFILES+("chess","shogi","janggi","xiangqi","makruk"))
    p.add_argument("--shards",type=int,default=12)
    p.add_argument("--games-per-shard",type=int,default=1024)
    a=p.parse_args()
    try:
        result=merge(a.input_root,a.output,a.variant,a.shards,a.games_per_shard)
        print(json.dumps(result,sort_keys=True),flush=True)
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
        print(f"Cannot combine unrelated native features: {exc}",file=sys.stderr)
        raise SystemExit(1)
if __name__=="__main__":main()
