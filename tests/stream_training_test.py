#!/usr/bin/env python3
"""Reproducibility, resumption, source integrity and >1M-row stream regression."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import numpy as np

TOOLS=Path(__file__).resolve().parents[1]/"tools"/"nnue"
sys.path.insert(0,str(TOOLS))
from stream_dataset import build_index, digest, open_db


def write_native_fixture(root, rows=360, games=36):
    root.mkdir(parents=True,exist_ok=True)
    if rows < games*2:
        raise ValueError("Need several positions per game")
    desc={"supported":True,"variant":"chess","format_version":1,"feature_version":1,
          "rules_sha256":"a"*64,"width":8,"height":8,"type_ids":[1,2,3,4,5,63],
          "input_dim":780,"hidden_dim":64,"qa":256,"qb":64}
    data=root/"data"
    data.mkdir()
    counts=[rows//games + (1 if g < rows%games else 0) for g in range(games)]
    with (data/"games.jsonl").open("w") as gf, (data/"positions.jsonl").open("w") as pf:
        serial=0
        for g,n in enumerate(counts):
            gid=hashlib.sha256(f"fixture-game:{g}".encode()).hexdigest()
            move_count=n+8
            record={"schema":1,"game_id":gid,"variant":"chess","rules_sha256":"a"*64,
                    "teacher_engine_sha256":"b"*64,"moves":["a2a3"]*move_count,
                    "plies":move_count,"rows":n,"truncated":True,"result_white":None}
            gf.write(json.dumps(record)+"\n")
            for j in range(n):
                # Two board features + one counted pocket feature; stable,
                # bounded, sparse, and more than 1M distinct possible inputs.
                first=serial%64
                second=65+((serial//64)%64)
                count=1+(serial//(64*64))
                if count > 512:
                    raise ValueError("Synthetic unique fixture capacity exhausted")
                features=[[first,1],[64,count],[second,1]]
                side=serial%2
                cp=(g*17+j*13)%401-200
                row={"schema":1,"game_id":gid,"ply":j+8,"fen":f"test-fixture-{serial}",
                     "variant":"chess","rules_sha256":"a"*64,"teacher_engine_sha256":"b"*64,
                     "side_to_move":side,"features":features,"teacher":
                     {"unit":"cp","value":cp,"bound":None,"depth":4,"nodes":100},
                     "label_cp_white":cp if side==0 else -cp}
                pf.write(json.dumps(row,separators=(",",":"))+"\n")
                serial+=1
    manifest={"schema":1,"generator_version":"svnnue-selfplay-v1","variant":"chess",
        "descriptor":desc,"rules_sha256":"a"*64,"engine_sha256":"b"*64,
        "source_commit":"c"*40,"source_working_tree_dirty":False,
        "teacher":{"engine_sha256":"b"*64,"score_perspective":"side_to_move",
                   "label_perspective":"white"},"game_count":games,"rows":rows,
        "files":{name:{"sha256":digest(data/name),"bytes":(data/name).stat().st_size}
                 for name in ("games.jsonl","positions.jsonl")}}
    (data/"manifest.json").write_text(json.dumps(manifest))
    return data


def execute(*args):
    p=subprocess.run([sys.executable,str(TOOLS/"train_stream.py"),*map(str,args)],
                     text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
    if p.returncode:
        raise AssertionError(f"Streaming trainer failed:\n{p.stdout[-4000:]}")
    return p.stdout


def normal_test(root):
    data=write_native_fixture(root)
    fresh=root/"fresh"
    resume=root/"resume"
    execute("--data",data,"--output",fresh,"--epochs",2,"--batch-size",32,"--seed",42)
    execute("--data",data,"--output",resume,"--epochs",2,"--batch-size",32,
            "--seed",42,"--stop-after-epoch",1)
    assert not (resume/"chess.svnnue").exists(), "Premature model export"
    execute("--data",data,"--output",resume,"--epochs",2,"--batch-size",32,
            "--seed",42,"--resume")
    # Exact optimizer moment, variance, parameter and chosen checkpoint identity.
    for prefix in ("weights","moment","variance","best"):
        with np.load(fresh/"state-epoch-00002.npz") as a, np.load(resume/"state-epoch-00002.npz") as b:
            for key in ("w1","b1","w2","b2"):
                assert np.array_equal(a[f"{prefix}_{key}"],b[f"{prefix}_{key}"]),f"Resume changed {prefix}_{key}"
    for filename in ("chess.svnnue","metrics.json","network-manifest.json"):
        assert digest(fresh/filename)==digest(resume/filename),f"Resume changed {filename}"
    original=build_index([data],fresh/"index.sqlite3",42)
    assert original["rows_read"]==360
    try:
        build_index([data],fresh/"index.sqlite3",43)
        raise AssertionError("Changed seed accepted")
    except ValueError: pass
    with (data/"positions.jsonl").open("a") as f:
        f.write("{}\n")
    try:
        build_index([data],fresh/"index.sqlite3",42)
        raise AssertionError("Corrupted source accepted")
    except ValueError: pass
    print("PASS deterministic whole-epoch Adam resume, model identity, cross-seed and corruption gates")


def stress_test(root, count):
    data=write_native_fixture(root, rows=count, games=512)
    dbpath=root/"massive.sqlite3"
    info=build_index([data],dbpath,20261026)
    assert info["rows_read"]==count
    assert info["rows_indexed"]>0
    with open_db(dbpath) as db:
        assert db.execute("SELECT COUNT(*) FROM examples").fetchone()[0]==info["rows_indexed"]
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps({"stress_rows_read":info["rows_read"],
                      "unique_training_inputs":info["rows_indexed"],
                      "peak_process_rss_kib":rss,"index_file_bytes":dbpath.stat().st_size}))
    assert rss<512*1024,f"Excessive indexer memory: {rss} KiB"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stress",action="store_true")
    p.add_argument("--rows",type=int,default=1000001)
    args=p.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        if args.stress:
            if args.rows<1000001: p.error("Stress test must exceed one million position rows")
            stress_test(root,args.rows)
        else:
            normal_test(root)


if __name__=="__main__":
    main()
