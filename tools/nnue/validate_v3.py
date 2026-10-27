#!/usr/bin/env python3
"""Real C++ inference parity + make/unmake gate for trained SVNNUE3.

Never report strength from numerical parity tests. The neural evaluation must
match the independent Python integer evaluator on heldout native positions.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from generate import Engine
from dataset_v3 import open_db,batches,sparse_batch,unpack
from model_v3 import digest,integer_predict,read_network

PREFIX="info string NNUE_FEATURES "

def json_reply(engine,command,prefix):
    engine.send(command)
    return json.loads(engine.until(lambda s:s.startswith(prefix))[-1][len(prefix):])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--engine",required=True,type=Path)
    p.add_argument("--model-dir",required=True,type=Path)
    p.add_argument("--positions",type=int,default=32)
    p.add_argument("--report",required=True,type=Path)
    a=p.parse_args()
    if not 8<=a.positions<=256:p.error("positions 8..256")
    modeldir=a.model_dir.resolve()
    summary=json.loads((modeldir/"index-summary.json").read_text())
    manifest=json.loads((modeldir/"network-manifest.json").read_text())
    metrics=json.loads((modeldir/"metrics.json").read_text())
    net_desc=manifest["descriptor"]
    native_desc=summary["descriptor"]
    compatible={key:value for key,value in net_desc.items() if key!="qa"}=={
        key:value for key,value in native_desc.items() if key!="qa"}
    if (manifest["dataset_signature"]!=summary["signature"]
        or metrics["dataset_signature"]!=summary["signature"]
        or not compatible or not manifest["quantization_passed"]):
        raise ValueError("Trained v3 model and source index mismatch")
    network=modeldir/manifest["network"]
    if digest(network)!=manifest["network_sha256"]:
        raise ValueError("Trained v3 payload checksum mismatch")
    q=read_network(network,net_desc)
    rows=[]
    games={}
    with open_db(modeldir/"index.sqlite3") as db:
        state=db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if state is None or state[0]!=summary["signature"]:
            raise ValueError("Heldout split index not bound to trained v3")
        for batch in batches(db,2,64,epoch_seed=0):
            for row in batch:
                rows.append(row)
                if len(rows)>=a.positions:break
            if len(rows)>=a.positions:break
        if len(rows)<8:raise ValueError("Insufficient native v3 heldout positions")
        for row in rows:
            if row[5] not in games:
                move_list=db.execute("SELECT moves FROM games WHERE game_id=?",
                                     (row[5],)).fetchone()
                if move_list is None:raise ValueError("Orphan heldout game ID")
                games[row[5]]=json.loads(move_list[0])
    w,b,side,_=sparse_batch(rows,summary["descriptor"]["input_dim"])
    expected=integer_predict(q,w,b,side)
    engine=Engine(a.engine.resolve(),60)
    checked=0
    searches=0
    failures=[]
    try:
        engine.option("UCI_Variant",summary["descriptor"]["variant"])
        engine.option("Threads",1)
        engine.option("Hash",16)
        engine.option("EvalFile",network.resolve())
        engine.option("EvalBackend","StocklessNNUEv3")
        status=json_reply(engine,"nnuestatus","info string NNUE_STATUS ")
        if (not status.get("active") or status.get("format")!=3
            or status.get("model_qa")!=net_desc["qa"]
            or status.get("model_qb")!=net_desc["qb"]):
            raise AssertionError("SVNNUE3 was not actually enabled")
        for i,row in enumerate(rows):
            game_id=row[5]
            ply=row[6]
            engine.position(games[game_id][:ply])
            native=json_reply(engine,"nnuefeaturesv3",PREFIX)
            v3=native["v3"]
            if (native["fen"]!=row[7] or native["side_to_move"]!=row[3]
                or v3["features_white"]!=[[idx,n] for idx,n in unpack(row[1])]
                or v3["features_black"]!=[[idx,n] for idx,n in unpack(row[2])]):
                raise AssertionError(f"Native full-history v3 position mismatch row {row[0]}")
            got=json_reply(engine,"nnueeval","info string NNUE_EVAL ")
            if (not got["used"] or not got["accumulator_matches"]
                or got["stm_cp"]!=int(expected[i])):
                raise AssertionError(f"C++ vs Python SVNNUE3 disagreement at row {row[0]}: {got}")
            checked+=1
            if searches<4 and not native["terminal"]["ended"]:
                bestmove,_=engine.search(3)
                if bestmove not in native["legal_moves"]:
                    raise AssertionError("Trained v3 search returned an illegal move")
                again=json_reply(engine,"nnueeval","info string NNUE_EVAL ")
                if not again["used"] or not again["accumulator_matches"]:
                    raise AssertionError("Trained v3 search mutated an accumulator incorrectly")
                searches+=1
        result={"schema":3,"variant":summary["descriptor"]["variant"],
                "network_sha256":digest(network),
                "engine_sha256":digest(a.engine),"passed":checked+searches,
                "integer_parity_checks":checked,"search_checks":searches,"failed":0,
                "elo":None,"strength_validated":False}
        a.report.parent.mkdir(parents=True,exist_ok=True)
        a.report.write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result))
    finally:
        engine.close()

if __name__=="__main__":main()
