#!/usr/bin/env python3
"""Recover 2026-10-27 original diverse SVNNUE3 corpus audit seed provenance.

Source generator finished all 60 shards; old compact audits accidentally
omitted the independently recorded manifest seeds. Recompute every audit
from its immutable original JSONL, compare the original audit except the
new seed field, and reject any source-manifest or fingerprint drift.
The input original self-play generation job does not need to be rerun.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"/"nnue"))
from verify_diverse_shard import verify
from audit_diverse_campaign import verify as verify_combined

PROFILES=("balanced","multipv","deep","exploration")
SOURCE_SHA="12063059cdabc355053f6c9373bfe9642858c63c"
ORIGINAL_RUN=40475358787


def run(input_root:Path,variant:str,out:Path,shards=12,min_unique=1000000):
    if shards!=12:
        raise ValueError("Recovery workflow is bound to twelve source shards")
    input_root=Path(input_root)
    recovered=[]
    for i in range(shards):
        folder=input_root/f"v3-diverse-data-{variant}-{i}"
        audit_file=folder/"audit.json"
        if not audit_file.is_file():
            raise ValueError(f"Missing original validated shard {variant}/{i}")
        previous=json.loads(audit_file.read_text(encoding="utf-8"))
        if "seed" in previous:
            raise ValueError("Source claimed old audit is missing seed but it was already included")
        manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        seed=20261027 + i*100003
        if type(manifest.get("seed")) is not int or manifest["seed"]!=seed:
            raise ValueError(f"{variant}/{i}: original manifest has wrong seed")
        if manifest.get("shard_id")!=i or manifest.get("profile")!=PROFILES[i//3]:
            raise ValueError(f"{variant}/{i}: original shard/profile mismatch")
        # Original raw JSONL and the SHA256-packed per-row fingerprint file
        # are re-read, not trusted simply because a GitHub job once passed.
        regenerated=verify(folder,variant,i,PROFILES[i//3],1024)
        if regenerated.get("seed")!=seed:
            raise ValueError("Verified audit seed did not match original manifest")
        # Counter() uses integer keys before serialization, whereas the
        # original audit was uploaded as JSON, which has string map keys.
        # Normalize BOTH structures to canonical JSON before comparison.
        without_seed=json.loads(json.dumps(
            {k:v for k,v in regenerated.items() if k!="seed"},
            sort_keys=True))
        if previous!=without_seed:
            changed=[k for k in sorted(set(previous)|set(without_seed))
                     if previous.get(k)!=without_seed.get(k)]
            details={k:{"original":str(previous.get(k))[:200],
                        "regenerated":str(without_seed.get(k))[:200]}
                     for k in changed}
            raise ValueError(f"{variant}/{i}: independently regenerated audit differs from original keys {details}")
        recovered.append({"shard":i,"seed":seed,
                          "source_manifest_sha256":regenerated["manifest_sha256"],
                          "raw_positions":regenerated["positions"]})
        print(json.dumps({"verified_original_shard":i,"variant":variant,
                          "seed":seed,"positions":regenerated["positions"]}),flush=True)
    # Reuse disk-backed full SHA256 deduplication. This step may reject a
    # legitimately generated corpus that did not meet the per-game target.
    proof=verify_combined(input_root,variant,shards,min_unique)
    if [p["seed"] for p in proof["source_shards"]]!=[x["seed"] for x in recovered]:
        raise ValueError("Cross-shard proof and source-manifest seed list disagree")
    proof.update({
        "recovery_schema":"svnnue3-recovered-original-audit-v1",
        "original_generation_run":ORIGINAL_RUN,
        "original_generation_source_sha":SOURCE_SHA,
        "reason":"original compact audit omitted manifest seed field",
        "source_manifests_reverified_from_raw_bytes":True,
        "recovered_source_seeds":recovered,
        "original_game_positions_regenerated":False
    })
    out=Path(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(proof,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"variant":variant,"real_generated_positions":proof["generated_positions"],
                      "unique_royal_inputs":proof["unique_model_inputs_sha256"],
                      "games":proof["games"],"reverified_shards":len(recovered),
                      "target_met":proof["unique_target_met"]}),flush=True)
    return proof


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--variant",required=True,choices=("chess","shogi","janggi","xiangqi","makruk"))
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--unique-goal",type=int,default=1000000)
    a=p.parse_args()
    try:run(a.source,a.variant,a.output,min_unique=a.unique_goal)
    except (ValueError,OSError,KeyError,TypeError,IndexError,json.JSONDecodeError) as err:
        print(f"REFUSE native-corpus recovery: {err}",file=sys.stderr)
        raise SystemExit(1)


if __name__=="__main__":main()
