#!/usr/bin/env python3
"""Same-build NNUE-v3 versus Classical UCI search throughput smoke.

NPS is not a pure neural-inference microbenchmark because evaluation
changes pruning/move ordering and hence searched node distribution.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"nnue"))
from generate import Engine,sha256


def options(engine, variant, network=None):
    engine.option("UCI_Variant",variant)
    engine.option("Hash",16)
    engine.option("Threads",1)
    engine.option("StocklessHybrid",True)
    engine.option("MultiPV",1)
    if network:
        engine.option("EvalFile",network)
        engine.option("EvalBackend","StocklessNNUE")
        engine.send("nnuestatus")
        status=json.loads(engine.until(lambda line:line.startswith("info string NNUE_STATUS "))[-1].split("NNUE_STATUS ",1)[1])
        if status.get("active") is not True or status.get("model_format_version")!=3:
            raise RuntimeError("V3 model did not activate")
    else:
        engine.option("EvalBackend","Classical")


def openings(engine,variant,count,ply,seed):
    result=[]
    for i in range(count):
        rng=random.Random(int.from_bytes(hashlib.sha256(f"{variant}-{seed}-{i}".encode()).digest()[:8],"big"))
        history=[]
        for step in range(ply):
            engine.position(history)
            pos=engine.features()
            if pos["terminal"]["ended"] or not pos["legal_moves"]:
                break
            history.append(rng.choice(sorted(pos["legal_moves"])))
        result.append(history)
    return result


def run_search(engine,moves,depth):
    engine.new_game()
    engine.position(moves)
    legal=engine.features()
    if legal["terminal"]["ended"]:
        return None
    engine.send(f"go depth {depth}")
    lines=engine.until(lambda line:line.startswith("bestmove "))
    best=lines[-1].split()[1]
    if best not in legal["legal_moves"]:
        raise RuntimeError("Engine returned illegal move")
    result=None
    for line in lines:
        if not line.startswith("info depth "):
            continue
        values={x:int(y) for x,y in re.findall(r"(depth|nodes|nps|time) (\d+)",line)}
        if values.get("depth")==depth and values.get("nodes",0)>0:
            result=values
    if not result:
        raise RuntimeError("Engine omitted completed depth/nodes")
    if result.get("nps",0)<=0 and result.get("time",0)>0:
        result["nps"]=int(result["nodes"]*1000/result["time"])
    result["bestmove"]=best
    result["moves"]=moves
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--engine",required=True,type=Path)
    p.add_argument("--network",required=True,type=Path)
    p.add_argument("--variant",required=True,choices=("chess","shogi","janggi","xiangqi","makruk"))
    p.add_argument("--depth",default=5,type=int)
    p.add_argument("--positions",default=8,type=int)
    p.add_argument("--opening-plies",default=8,type=int)
    p.add_argument("--seed",default=20261030,type=int)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if not (3<=a.depth<=8 and 2<=a.positions<=50 and 0<=a.opening_plies<=24):
        p.error("Invalid depth, positions or opening plies")
    eng=a.engine.resolve()
    net=a.network.resolve()
    classic=Engine(eng,60)
    neural=Engine(eng,60)
    try:
        options(classic,a.variant)
        options(neural,a.variant,net)
        start=openings(classic,a.variant,a.positions,a.opening_plies,a.seed)
        reports=[]
        for tag,engine in (("classical",classic),("svnnue-v3",neural)):
            rows=[run_search(engine,moves,a.depth) for moves in start]
            accepted=[r for r in rows if r is not None]
            if len(accepted)<2:raise RuntimeError("Too few nonterminal benchmark positions")
            reports.append({"backend":tag,"searches":accepted,
                            "median_nps":statistics.median(r.get("nps",0) for r in accepted),
                            "total_nodes":sum(r["nodes"] for r in accepted),
                            "total_ms":sum(r.get("time",0) for r in accepted)})
        base,recent=reports
        ratio=(recent["median_nps"]/base["median_nps"]) if base["median_nps"] else None
        output={"variant":a.variant,"depth":a.depth,"positions":a.positions,
                "opening_plies":a.opening_plies,"engine_sha256":sha256(eng),
                "network_sha256":sha256(net),"median_nps_ratio":ratio,
                "results":reports,"strength_validated":False,"elo":None,
                "limitation":"Search NPS also depends on selected position/node mix; not pure neural inference throughput or strength."}
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(output,indent=2)+"\n")
        print(json.dumps({k:v for k,v in output.items() if k!="results"}))
    finally:
        classic.close();neural.close()


if __name__=="__main__":
    main()
