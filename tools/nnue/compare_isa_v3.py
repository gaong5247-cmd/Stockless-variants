#!/usr/bin/env python3
"""Independent scalar/AVX2 SVNNUE3 integer/feature/search parity gate.

Replays frozen *real* heldout games in two separately compiled engines. Both
implementations must agree with each other AND native dataset features.
NPS is deliberately measured in a separate workflow step, never conflated
with exact numerical correctness.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from dataset_v3 import batches, open_db, unpack
from generate import Engine, sha256
from model_v3 import read_network, digest
from validate_v3 import json_reply

PREFIX="info string NNUE_FEATURES "

def engine_with_model(path:Path,model:Path,variant:str,implementation:str)->tuple[Engine,dict]:
    engine=Engine(path.resolve(),timeout=90)
    try:
        engine.option("UCI_Variant",variant)
        engine.option("Threads",1)
        engine.option("Hash",16)
        engine.option("MultiPV",1)
        engine.option("StocklessHybrid",True)
        engine.option("Use NNUE",False)
        engine.option("EvalFile",model.resolve())
        engine.option("EvalBackend","StocklessNNUEv3")
        status=json_reply(engine,"nnuestatus","info string NNUE_STATUS ")
        if (status.get("active") is not True or status.get("format")!=3
            or status.get("model_variant")!=variant
            or status.get("accumulator_implementation")!=implementation):
            raise AssertionError(
                f"{implementation} engine did not activate its distinct compiled path: {status}"
            )
        return engine,status
    except BaseException:
        engine.close()
        raise

def run(args):
    summary=json.loads((args.model_dir/"model/index-summary.json").read_text())
    manifest=json.loads((args.model_dir/"model/network-manifest.json").read_text())
    variant=summary["descriptor"]["variant"]
    network=args.model_dir/"model"/(variant+".svnnue")
    if (manifest["network_sha256"]!=digest(network)
        or summary["descriptor"]!=manifest["descriptor"]
        or manifest["dataset_signature"]!=summary["signature"]):
        raise ValueError("Trained network and heldout game index mismatch")
    read_network(network,summary["descriptor"])
    selected=[]
    games={}
    with open_db(args.model_dir/"model/index.sqlite3") as db:
        row=db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if not row or row[0]!=summary["signature"]:
            raise ValueError("Source SQLite split does not belong to this model")
        for block in batches(db,2,64,epoch_seed=args.seed):
            for item in block:
                selected.append(item)
                if len(selected)>=args.positions:break
            if len(selected)>=args.positions:break
        for rec in selected:
            if rec[5] not in games:
                row=db.execute("SELECT moves FROM games WHERE game_id=?",(rec[5],)).fetchone()
                if not row:raise ValueError("Orphan validation game")
                games[rec[5]]=json.loads(row[0])
    if len(selected)<8:
        raise ValueError("Too few independent heldout samples")
    scalar,scalar_status=engine_with_model(args.scalar,network,variant,"scalar")
    try:
        avx,avx_status=engine_with_model(args.avx2,network,variant,"avx2")
    except BaseException:
        scalar.close()
        raise
    begun=time.monotonic()
    checked=0
    searches=0
    try:
        for pos in selected:
            ident,features_white,features_black,side,_label,gid,ply,fen=pos
            moves=games[gid][:ply]
            probes=[]
            states=[]
            for engine in (scalar,avx):
                engine.position(moves)
                native=json_reply(engine,"nnuefeaturesv3",PREFIX)
                got=json_reply(engine,"nnueeval","info string NNUE_EVAL ")
                if native.get("fen")!=fen or native.get("side_to_move")!=side:
                    raise AssertionError(f"Game replay failure: {variant}/{ident}")
                if (not got.get("used") or got.get("format")!=3
                    or got.get("accumulator_matches") is not True):
                    raise AssertionError(f"Native cached/full-refresh mismatch: {variant}/{ident}")
                v3=native["v3"]
                if (v3.get("features_white")!=[[int(i),int(v)] for i,v in unpack(features_white)]
                    or v3.get("features_black")!=[[int(i),int(v)] for i,v in unpack(features_black)]):
                    raise AssertionError(f"Native v3 feature mismatch: {variant}/{ident}")
                states.append(native)
                probes.append(got)
            if (states[0]!=states[1] or
                any(probes[0].get(k)!=probes[1].get(k)
                    for k in ("white_cp","stm_cp","used","accumulator_matches"))):
                raise AssertionError(f"AVX2 vs scalar integer divergence: {variant}/{ident}")
            checked+=1
            if searches<args.search_positions and not states[0]["terminal"]["ended"]:
                legal=states[0]["legal_moves"]
                best=[]
                for engine in (scalar,avx):
                    move,_=engine.search(args.depth)
                    if move not in legal:
                        raise AssertionError(f"Illegal depth-{args.depth} search move")
                    again=json_reply(engine,"nnueeval","info string NNUE_EVAL ")
                    if (not again["used"] or not again["accumulator_matches"]
                        or again["stm_cp"]!=probes[0]["stm_cp"]):
                        raise AssertionError("Search make/unmake corrupted v3 root accumulators")
                    best.append(move)
                # The chosen move is diagnostic (search tie-breaking may differ).
                searches+=1
        report={"schema":3,"variant":variant,
            "model_sha256":digest(network),
            "dataset_signature":summary["signature"],
            "scalar_engine_sha256":sha256(args.scalar),
            "avx2_engine_sha256":sha256(args.avx2),
            "scalar_backend":scalar_status["accumulator_implementation"],
            "avx2_backend":avx_status["accumulator_implementation"],
            "positions_checked":checked,"search_positions_checked":searches,
            "two_engine_integer_disagreements":0,
            "source_game_feature_mismatches":0,
            "search_return_state_errors":0,
            "elapsed_seconds":round(time.monotonic()-begun,3),
            "elo":None,"strength_validated":False}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        print(json.dumps(report,sort_keys=True),flush=True)
    finally:
        scalar.close()
        avx.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scalar",required=True,type=Path)
    p.add_argument("--avx2",required=True,type=Path)
    p.add_argument("--model-dir",required=True,type=Path)
    p.add_argument("--positions",type=int,default=64)
    p.add_argument("--search-positions",type=int,default=6)
    p.add_argument("--depth",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261027)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if not (8<=a.positions<=256 and 1<=a.search_positions<=a.positions
            and 1<=a.depth<=8 and 0<=a.seed<2**32):
        p.error("Invalid bounded cross-ISA comparison settings")
    run(a)

if __name__=="__main__":
    main()
