#!/usr/bin/env python3
"""Native two-royal-perspective selfplay teacher data for SVNNUE3.

V3 inputs are always exported by the actual C++ engine, not reinterpreted
from variant-specific FEN or Python piece notation. No fake game results.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from generate import Engine, canonical, sha256, score_label, validate_state
from model_v3 import validate_descriptor

PREFIX="info string NNUE_FEATURES "
GENERATOR="svnnue3-native-selfplay-1"

def native_state(engine,variant,expected=None):
    engine.send("nnuefeaturesv3")
    response=engine.until(lambda line:line.startswith(PREFIX))[-1]
    state=json.loads(response[len(PREFIX):])
    validate_state(state,variant)
    v3=state.get("v3")
    if not isinstance(v3,dict) or v3.get("schema")!=3 or "error" in v3:
        raise ValueError("Native C++ v3 feature command failed: "+str(v3))
    desc=v3["descriptor"]
    validate_descriptor(desc)
    if desc["variant"]!=variant:
        raise ValueError("Wrong v3 variant identity")
    if expected is not None and desc!=expected:
        raise ValueError("Native v3 descriptor changed during dataset")
    for perspective in ("white","black"):
        features=v3[f"features_{perspective}"]
        if not isinstance(features,list) or not features:
            raise ValueError("Missing royal-oriented feature vector")
        previous=-1
        mass=0
        for pair in features:
            if (not isinstance(pair,list) or len(pair)!=2 or
                any(type(x) is not int for x in pair)):
                raise ValueError("Malformed native v3 sparse feature")
            idx,count=pair
            if idx<=previous or idx>=desc["input_dim"] or count<1:
                raise ValueError("Out of range or unsorted v3 feature")
            previous=idx
            mass+=count
        if mass>512:raise ValueError("Native v3 feature mass unsafe")
    if v3["side_to_move"]!=state["side_to_move"]:
        raise ValueError("v3 and base rules disagree on side to move")
    return state,v3

def generate(args):
    path=args.engine.resolve()
    if not path.is_file():raise ValueError("Missing native teacher engine")
    dest=args.output.resolve()
    if dest.exists() and any(dest.iterdir()):
        raise ValueError("Refusing to overwrite an existing dataset")
    dest.mkdir(parents=True,exist_ok=True)
    teacher_hash=sha256(path)
    engine=Engine(path,args.timeout)
    games_path=dest/"games.jsonl"
    positions_path=dest/"positions.jsonl"
    positions=0
    complete=0
    censored=0
    expected=None
    options={"UCI_Variant":args.variant,"Threads":1,"Hash":16,"MultiPV":1,
             "EvalBackend":"Classical","Use NNUE":False,"StocklessHybrid":True}
    try:
        for key,value in options.items():engine.option(key,value)
        with games_path.open("x",encoding="utf-8") as games,positions_path.open("x",encoding="utf-8") as records:
            for i in range(args.games):
                seed=int.from_bytes(hashlib.sha256(
                    f"{GENERATOR}:{args.variant}:{args.seed}:{i}".encode()).digest()[:8],"big")
                rng=random.Random(seed)
                moves=[]
                rows=[]
                engine.new_game()
                for ply in range(args.max_plies+1):
                    engine.position(moves)
                    state,v3=native_state(engine,args.variant,expected)
                    if expected is None:expected=v3["descriptor"]
                    if state["terminal"]["ended"] or ply==args.max_plies:
                        break
                    if ply<args.opening_plies:
                        moves.append(rng.choice(sorted(state["legal_moves"])))
                        continue
                    move,teacher=engine.search(args.depth)
                    if move not in state["legal_moves"]:
                        raise ValueError(f"Native teacher illegal move: {move}")
                    if teacher is not None:
                        cp=score_label(teacher,state["side_to_move"])
                        rows.append({"schema":3,"ply":ply,"fen":state["fen"],
                                     "side_to_move":state["side_to_move"],
                                     "teacher_white_cp":cp,
                                     "teacher_stm_cp":cp*(1 if state["side_to_move"]==0 else -1),
                                     "teacher":teacher,
                                     "features_white":v3["features_white"],
                                     "features_black":v3["features_black"],
                                     "white_bucket":v3["white_bucket"],
                                     "black_bucket":v3["black_bucket"]})
                    moves.append(move)
                gid=hashlib.sha256(canonical({
                    "generator":GENERATOR,"game":i,"variant":args.variant,
                    "source_engine":teacher_hash,"seed":seed,
                    "depth":args.depth,"maximum":args.max_plies
                }).encode()).hexdigest()
                ended=state["terminal"]["ended"]
                outcome=(state["terminal"]["result_stm"]*
                         (1 if state["side_to_move"]==0 else -1)) if ended else None
                game={"schema":3,"game_id":gid,"seed":seed,"variant":args.variant,
                      "moves":moves,"plies":len(moves),"rows":len(rows),
                      "result_white":outcome,"truncated":not ended,
                      "terminal_reason":state["terminal"]["reason"] if ended else "max_plies",
                      "final_fen":state["fen"]}
                games.write(canonical(game)+"\n")
                for record in rows:
                    records.write(canonical({**record,"game_id":gid})+"\n")
                positions+=len(rows)
                complete+=int(ended)
                censored+=int(not ended)
                print(json.dumps({"variant":args.variant,"game":i+1,
                                  "total_games":args.games,"positions":positions,
                                  "completed":complete,"censored":censored}),flush=True)
        if not expected or positions<16:
            raise ValueError("Insufficient native v3 learning positions")
        manifest={"schema":3,"generator":GENERATOR,"descriptor":expected,
                  "variant":args.variant,"seed":args.seed,
                  "native_engine_sha256":teacher_hash,"teacher_options":options,
                  "games":args.games,"positions":positions,
                  "completed_games":complete,"censored_games":censored,
                  "teacher_depth":args.depth,"max_plies":args.max_plies,
                  "opening_plies":args.opening_plies,
                  "score_policy":"exact main PV, native value white and side-to-move cp",
                  "split_policy":"whole-game hash-based split, deduplicate dual-perspective inputs",
                  "files":{name:{"sha256":sha256(dest/name),
                                 "bytes":(dest/name).stat().st_size}
                           for name in ("games.jsonl","positions.jsonl")}}
        (dest/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        print(json.dumps({"v3_positions":positions,"native_games":args.games,
                          "manifest":str(dest/"manifest.json")}))
        return manifest
    finally:
        engine.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine",type=Path,required=True)
    p.add_argument("--variant",choices=("chess","shogi","janggi","xiangqi","makruk"),required=True)
    p.add_argument("--games",type=int,default=32)
    p.add_argument("--max-plies",type=int,default=96)
    p.add_argument("--depth",type=int,default=2)
    p.add_argument("--opening-plies",type=int,default=8)
    p.add_argument("--seed",type=int,default=20261027)
    p.add_argument("--timeout",type=int,default=60)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if not (6<=a.games<=1024 and 32<=a.max_plies<=256 and
            1<=a.depth<=8 and 0<=a.opening_plies<=24 and
            0<=a.seed<2**32 and 1<=a.timeout<=300):
        p.error("Unsupported bounded v3 data generation budget")
    try:generate(a)
    except Exception as exc:
        print("v3 native data generator failed:",exc,file=sys.stderr)
        raise

if __name__=="__main__": main()
