#!/usr/bin/env python3
"""Paired search-throughput diagnostic: Classical vs native SVNNUE3.

Equal hardware, engine binary, threads, hash, and opening position. Separate
search trees mean NPS ratios are NOT a pure evaluator speed benchmark.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"nnue"))
from generate import Engine, sha256

def best_depth(lines,depth):
    observed=[]
    for line in lines:
        if not line.startswith("info "):continue
        d=re.search(r"\bdepth (\d+)\b",line)
        n=re.search(r"\bnodes (\d+)\b",line)
        t=re.search(r"\btime (\d+)\b",line)
        if d and n and t and int(d[1])==depth:
            observed.append((int(n[1]),int(t[1])))
    if not observed:return {"nodes":None,"time_ms":None,"nps":None}
    nodes,ms=observed[-1]
    return {"nodes":nodes,"time_ms":ms,"nps":(1000*nodes/ms if ms else None)}

def search(engine,moves,depth):
    engine.position(moves)
    engine.send(f"go depth {depth}")
    lines=engine.until(lambda s:s.startswith("bestmove "))
    return best_depth(lines,depth)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--engine",type=Path,required=True)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--variant",required=True,
                        choices=("chess","shogi","janggi","xiangqi","makruk"))
    parser.add_argument("--depth",type=int,default=7)
    parser.add_argument("--positions",type=int,default=6)
    parser.add_argument("--seed",type=int,default=20261027)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    if not 4<=a.depth<=10 or not 2<=a.positions<=24:
        parser.error("Unsupported fixed-depth benchmark")
    engine=a.engine.resolve()
    model=a.model.resolve()
    if not engine.is_file() or not model.is_file():raise ValueError("Engine/network missing")
    classical=Engine(engine,90)
    nnue=Engine(engine,90)
    records=[]
    try:
        for obj in (classical,nnue):
            obj.option("UCI_Variant",a.variant)
            obj.option("Threads",1)
            obj.option("Hash",16)
            obj.option("MultiPV",1)
            obj.option("StocklessHybrid",True)
            obj.option("Use NNUE",False)
        nnue.option("EvalFile",model)
        nnue.option("EvalBackend","StocklessNNUEv3")
        nnue.send("nnuestatus")
        status=json.loads(nnue.until(lambda s:s.startswith("info string NNUE_STATUS "))[-1].split("NNUE_STATUS ",1)[1])
        if status.get("active") is not True or status.get("format")!=3:
            raise ValueError("v3 not actually loaded/active")
        for index in range(a.positions):
            seed=int(hashlib.sha256(f"{a.seed}:{a.variant}:{index}".encode()).hexdigest()[:16],16)
            rng=random.Random(seed)
            moves=[]
            classical.new_game()
            for ply in range(3+index%5):
                classical.position(moves)
                state=classical.features()
                if state["terminal"]["ended"]:break
                moves.append(rng.choice(sorted(state["legal_moves"])))
            classical.new_game()
            nnue.new_game()
            c=search(classical,moves,a.depth)
            n=search(nnue,moves,a.depth)
            records.append({"opening_moves":moves,"classical":c,"nnue_v3":n})
            print(json.dumps({"variant":a.variant,"position":index,
                              "classical":c,"nnue_v3":n}),flush=True)
        # Aggregate nodes/time, not average of individual NPS measurements.
        def metric(which):
            valid=[r[which] for r in records if r[which]["time_ms"]]
            nodes=sum(x["nodes"] for x in valid)
            ms=sum(x["time_ms"] for x in valid)
            return {"measured_positions":len(valid),"nodes":nodes,
                    "time_ms":ms,"nps":nodes*1000/ms if ms else None}
        report={"schema":3,"variant":a.variant,"engine_sha256":sha256(engine),
                "network_sha256":sha256(model),"depth":a.depth,
                "classical":metric("classical"),"nnue_v3":metric("nnue_v3"),
                "note":"This is equal-depth search throughput; search trees may differ. Not isolated eval microbenchmark or Elo.",
                "records":records}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k!="records"}))
    finally:
        classical.close()
        nnue.close()

if __name__=="__main__":main()
