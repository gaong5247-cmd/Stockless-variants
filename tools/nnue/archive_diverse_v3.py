#!/usr/bin/env python3
"""Package previously CERTIFIED, real native v3 shards into release ZIPs.

Never publish data from an unverified generator artifact. Independently
rehashes source and checks every model-input fingerprint and game history
before copying into durable per-variant archives.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile,ZIP_DEFLATED

from verify_diverse_shard import verify

PROFILES=("balanced","multipv","deep","exploration")

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def archive_shards(root,proof_file,variant,output):
    proof=json.loads(Path(proof_file).read_text())
    if (proof.get("schema")!=3 or proof.get("variant")!=variant
        or proof.get("shards")!=12 or proof.get("unique_target_met") is not True
        or proof.get("unique_model_inputs_sha256",0)<1000000):
        raise ValueError("This variant has not independently certified one million distinct inputs")
    audited={s["id"]:s for s in proof["source_shards"]}
    if set(audited)!=set(range(12)):
        raise ValueError("Expected all 12 certified native data shards")
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Publish staging must be empty to prevent stale release assets")
    output.mkdir(parents=True,exist_ok=True)
    assets=[]
    count=0
    for i in range(12):
        folder=root/f"v3-diverse-data-{variant}-{i}"
        profile=PROFILES[i//3]
        if not folder.is_dir():
            raise ValueError(f"Missing original source data artifact: {folder}")
        result=verify(folder,variant,i,profile,1024)
        expected=audited[i]
        for key,source in (("manifest_sha256","manifest_sha256"),
                           ("fingerprints_sha256","files")):
            wanted=expected[key]
            actual=result[source] if source!="files" else result["files"]["fingerprints.sha256.bin"]["sha256"]
            if wanted!=actual:
                raise ValueError(f"Provenance differs from accepted source audit: {i}/{key}")
        if (result["positions"]!=expected["positions"]
            or result["games"]!=1024
            or result["source_engine_sha256"]!=proof["native_teacher_sha256"]
            or result["rules_sha256"]!=proof["native_rules_sha256"]):
            raise ValueError(f"Dataset diverged from successful acceptance proof: shard {i}")
        archive_name=f"svnnue3-{variant}-diverse-{i:02d}-{profile}.zip"
        destination=output/archive_name
        with ZipFile(destination,"w",ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for name in ("games.jsonl","positions.jsonl","fingerprints.sha256.bin",
                         "manifest.json","audit.json"):
                z.write(folder/name,name)
        # Archive must decode correctly. The inner files are already SHA
        # checked above and the entire outer ZIP gets a new digest.
        with ZipFile(destination) as z:
            if z.testzip() is not None:
                raise ValueError(f"Corrupt native data release ZIP: {archive_name}")
        count+=result["positions"]
        assets.append({"name":archive_name,"bytes":destination.stat().st_size,
                       "sha256":sha(destination),"shard":i,
                       "profile":profile,"positions":result["positions"]})
    if count!=proof["generated_positions"]:
        raise ValueError("Archived positions do not match the independently accepted proof")
    (output/"SHASUMS256.txt").write_text("".join(
        a["sha256"]+"  "+a["name"]+"\n" for a in assets),encoding="utf-8")
    release_manifest={
        "schema":3,"variant":variant,"experiment":"svnnue3-diverse-20261027",
        "source_revision":"12063059cdabc355053f6c9373bfe9642858c63c",
        "source_actions_run":40475358787,"source_teacher_sha256":proof["native_teacher_sha256"],
        "rules_sha256":proof["native_rules_sha256"],
        "generated_positions":count,
        "unique_native_input_fingerprints":proof["unique_model_inputs_sha256"],
        "native_games":proof["games"],"profiles_games":proof["profiles_games"],
        "source_shards":proof["source_shards"],
        "files":assets,
        "data_generated_and_audited":True,
        "nnue_network_trained":False,
        "elo_claim":None
    }
    (output/"CORPUS_MANIFEST.json").write_text(
        json.dumps(release_manifest,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (output/"CORPUS_PROOF.json").write_text(
        json.dumps(proof,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return release_manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--proof",type=Path,required=True)
    p.add_argument("--variant",required=True,
                   choices=("chess","shogi","janggi","xiangqi","makruk"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=archive_shards(args.root,args.proof,args.variant,args.output)
    print(json.dumps({"variant":result["variant"],"games":result["native_games"],
        "generated":result["generated_positions"],
        "unique":result["unique_native_input_fingerprints"],
        "files":len(result["files"]),"ready":True}),flush=True)

if __name__=="__main__":main()
