#!/usr/bin/env python3
"""Native heldout parity and legal-move gate for streaming SVNNUE models."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import sqlite3
import numpy as np
from model import digest, read_network, integer_predict
from generate import Engine
from stream_dataset import open_db, dense_batch


def run(engine_path,index,model_dir,positions,seed,report):
    summary=json.loads((model_dir/"index-summary.json").read_text(encoding="utf-8"))
    manifest=json.loads((model_dir/"network-manifest.json").read_text(encoding="utf-8"))
    metrics=json.loads((model_dir/"metrics.json").read_text(encoding="utf-8"))
    if (manifest["stream_index_signature"]!=summary["signature"] or
        metrics["stream_index_signature"]!=summary["signature"] or
        manifest["descriptor"]!=summary["descriptor"] or
        not manifest["quantization_passed"]):
        raise ValueError("Training and model provenance do not match")
    network=model_dir/manifest["network"]
    if digest(network)!=manifest["network_sha256"]:
        raise ValueError("SVNNUE file checksum mismatch")
    weights=read_network(network,summary["descriptor"])
    with open_db(index) as db:
        sig=db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if sig is None or sig[0]!=summary["signature"]:
            raise ValueError("Index does not match model's dataset")
        lo,hi,n=db.execute("SELECT MIN(id),MAX(id),COUNT(*) FROM examples WHERE partition=2").fetchone()
        if not n:
            raise ValueError("No heldout positions")
        rng=random.Random(seed)
        selected=[]
        seen=set()
        for j in range(min(n,positions)*10):
            if len(selected)>=min(n,positions):
                break
            pivot=rng.randint(lo,hi)
            row=db.execute("SELECT id,features,side,label,game_id,ply,fen FROM examples "
                           "WHERE partition=2 AND id>=? ORDER BY id LIMIT 1",(pivot,)).fetchone()
            if row is not None and row[0] not in seen:
                selected.append(row)
                seen.add(row[0])
        if len(selected)<min(positions,n):
            rows=db.execute("SELECT id,features,side,label,game_id,ply,fen "
                            "FROM examples WHERE partition=2 ORDER BY id LIMIT ?",(positions,)).fetchall()
            for row in rows:
                if row[0] not in seen:
                    selected.append(row)
                    seen.add(row[0])
                if len(selected)>=min(n,positions):
                    break
        if not selected:
            raise ValueError("No selected heldout records")
        x,sides,_=dense_batch(selected,summary["descriptor"]["input_dim"])
        expected=integer_predict(weights,x.astype(np.int64),sides)
        games={}
        for row in selected:
            gid=row[4]
            if gid not in games:
                games[gid]=json.loads(db.execute("SELECT moves FROM games WHERE game_id=?",(gid,)).fetchone()[0])
    engine=Engine(engine_path.resolve(),60)
    result={"schema":1,"variant":summary["variant"],"network_sha256":digest(network),
            "engine_sha256":digest(engine_path),"stream_index_signature":summary["signature"],
            "checks":0,"failed":0,"positions_checked":0,"searches_checked":0,
            "elo":None,"strength_validated":False}
    try:
        engine.option("UCI_Variant",summary["variant"])
        engine.option("Threads",1)
        engine.option("Hash",16)
        engine.option("EvalFile",network.resolve())
        engine.option("EvalBackend","StocklessNNUE")
        engine.send("nnuestatus")
        raw=engine.until(lambda line:line.startswith("info string NNUE_STATUS "))[-1]
        status=json.loads(raw.split("NNUE_STATUS ",1)[1])
        if not status["active"] or status["model_variant"]!=summary["variant"]:
            raise AssertionError("Trained model was not activated")
        for i,row in enumerate(selected):
            engine.position(games[row[4]][:row[5]])
            native=engine.features()
            # Tensor parity alone would miss incorrect game replay.
            if (native["fen"]!=row[6] or native["side_to_move"]!=row[2] or
                native["features"]!=[[int(a),int(b)] for a,b in
                    __import__("stream_dataset").decode_features(row[1])]):
                raise AssertionError(f"Native position/features differ at row {row[0]}")
            engine.send("nnueeval")
            check=json.loads(engine.until(lambda l:l.startswith("info string NNUE_EVAL "))[-1]
                             .split("NNUE_EVAL ",1)[1])
            if not check["used"] or not check["accumulator_matches"] or check["white_cp"]!=int(expected[i]):
                raise AssertionError(f"NNUE integer or accumulator mismatch at {row[0]}")
            result["checks"]+=1
            result["positions_checked"]+=1
            if result["searches_checked"]<4 and not native["terminal"]["ended"]:
                best,_=engine.search(3)
                if best not in native["legal_moves"]:
                    raise AssertionError("NNUE search returned an illegal move")
                result["checks"]+=1
                result["searches_checked"]+=1
        return result
    finally:
        engine.close()
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--engine",type=Path,required=True)
    p.add_argument("--index",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--positions",type=int,default=32)
    p.add_argument("--seed",type=int,default=20261026)
    a=p.parse_args()
    if not 1<=a.positions<=512: p.error("positions must be 1..512")
    result=run(a.engine,a.index,a.model_dir,a.positions,a.seed,a.report)
    print(json.dumps(result))
if __name__=="__main__":
    main()
