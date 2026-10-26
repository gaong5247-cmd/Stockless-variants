#!/usr/bin/env python3
"""Auditable paired Classical-vs-SVNNUE games with native adjudication only.

A max-ply stop is *censored*, not a draw. No invented Elo. Games alternate
colors and share the same deterministic seeded opening in each paired match.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from generate import Engine, sha256


def id_sha(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def one_game(classical,neural,opening,variant,nnue_white,depth,max_plies):
    moves=opening.copy()
    # Never assume WHITE moves first. Some regional variants use a different
    # native starting side, and opening parity alone is not proof of color.
    while len(moves)<max_plies:
        classical.position(moves)
        judge=classical.features()
        if judge["terminal"]["ended"]:
            break
        nnue_to_move=(judge["side_to_move"]==0)==nnue_white
        active=neural if nnue_to_move else classical
        if nnue_to_move:
            neural.position(moves)
            state=neural.features()
            if (state["side_to_move"]!=judge["side_to_move"] or
                    state["fen"]!=judge["fen"] or
                    state["legal_moves"]!=judge["legal_moves"]):
                raise RuntimeError("Neural/classical rule state diverged before move")
        else:
            state=judge
        move,_=active.search(depth)
        if move not in state["legal_moves"]:
            raise RuntimeError(f"Illegal {move} from {'NNUE' if nnue_to_move else 'Classical'}")
        moves.append(move)
    # Native check even when max_plies reached: final move could be decisive.
    classical.position(moves)
    ending=classical.features()
    neural.position(moves)
    parallel=neural.features()
    if (ending["fen"]!=parallel["fen"] or
        ending["terminal"]!=parallel["terminal"] or
        ending["legal_moves"]!=parallel["legal_moves"]):
        raise RuntimeError("Classical and NNUE native rule state diverged")
    completed=ending["terminal"]["ended"]
    white_result= (ending["terminal"]["result_stm"] *
                  (1 if ending["side_to_move"]==0 else -1)) if completed else None
    nnue_result=white_result*(1 if nnue_white else -1) if completed else None
    return {"moves":moves,"plies":len(moves),"completed":completed,
            "censored":not completed,"result_white":white_result,
            "result_nnue":nnue_result,
            "termination":ending["terminal"]["reason"] if completed else "max_plies",
            "final_fen":ending["fen"]}


def run(args):
    engine=args.engine.resolve()
    model=args.model.resolve()
    if not engine.exists() or not model.exists():
        raise ValueError("Engine and model must exist")
    classical=Engine(engine,args.timeout)
    neural=Engine(engine,args.timeout)
    results=[]
    try:
        for process in (classical,neural):
            process.option("UCI_Variant",args.variant)
            process.option("Threads",1)
            process.option("Hash",16)
            process.option("MultiPV",1)
            process.option("StocklessHybrid",True)
            process.option("Use NNUE",False)
        neural.option("EvalFile",model)
        neural.option("EvalBackend","StocklessNNUE")
        neural.send("nnuestatus")
        active=json.loads(neural.until(lambda line:line.startswith("info string NNUE_STATUS "))[-1]
                          .split("NNUE_STATUS ",1)[1])
        if (active.get("active") is not True or active.get("model_variant")!=args.variant):
            raise RuntimeError("Model is not active for this variant")
        for pair in range(args.pairs):
            classical.new_game();neural.new_game()
            rng=random.Random(int(id_sha({"seed":args.seed,"pair":pair,"variant":args.variant})[:16],16))
            opening=[]
            classical.position([])
            for k in range(args.opening_plies):
                classical.position(opening)
                state=classical.features()
                if state["terminal"]["ended"]:
                    break
                opening.append(rng.choice(sorted(state["legal_moves"])))
            for nnue_white in (True,False):
                classical.new_game();neural.new_game()
                record=one_game(classical,neural,opening,args.variant,nnue_white,
                                args.depth,args.max_plies)
                record.update(pair=pair,nnue_white=nnue_white,opening=opening)
                results.append(record)
                print(json.dumps({"pair":pair,"nnue_white":nnue_white,
                                  "result":record["result_nnue"],
                                  "completed":record["completed"],"plies":record["plies"]}),
                      flush=True)
        wins=sum(r["result_nnue"]==1 for r in results)
        draws=sum(r["result_nnue"]==0 for r in results)
        losses=sum(r["result_nnue"]==-1 for r in results)
        censored=sum(r["result_nnue"] is None for r in results)
        played=wins+losses+draws
        score=(wins+draws*.5)/played if played else None
        # A score interval from completed games is descriptive only, because
        # early censored games may bias completion. No Elo transformation.
        interval=None
        if played>=2:
            # Wilson-style interval avoids the spuriously zero-width normal
            # interval for all-loss or all-win pilot matches. With draws
            # scored 0.5 this is descriptive, not a binomial Elo CI.
            z=1.96
            p=score
            denominator=1+z*z/played
            center=(p+z*z/(2*played))/denominator
            margin=z*math.sqrt(p*(1-p)/played+z*z/(4*played*played))/denominator
            interval=[max(0.,center-margin),min(1.,center+margin)]
        summary={"schema":1,"variant":args.variant,"engine_sha256":sha256(engine),
                 "network_sha256":sha256(model),"depth":args.depth,"pairs":args.pairs,
                 "games":len(results),"max_plies":args.max_plies,"seed":args.seed,
                 "wins":wins,"draws":draws,"losses":losses,"censored":censored,
                 "score_completed_only":score,"wilson_style_95pct_interval_completed_only":interval,
                 "uncertainty":"Censoring is not a draw; interval ignores censored games and is not an Elo CI",
                 "strength_validated":False,"elo":None,"games_records":results}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        print(json.dumps({k:v for k,v in summary.items() if k!="games_records"}))
        return summary
    finally:
        classical.close();neural.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--engine",required=True,type=Path)
    p.add_argument("--model",required=True,type=Path)
    p.add_argument("--variant",choices=("chess","shogi","janggi","xiangqi","makruk"),required=True)
    p.add_argument("--pairs",type=int,default=8)
    p.add_argument("--opening-plies",type=int,default=4)
    p.add_argument("--depth",type=int,default=3)
    p.add_argument("--max-plies",type=int,default=240)
    p.add_argument("--timeout",type=float,default=60)
    p.add_argument("--seed",type=int,default=20261026)
    p.add_argument("--output",required=True,type=Path)
    a=p.parse_args()
    if not (1<=a.pairs<=2048 and 0<=a.opening_plies<=20 and 1<=a.depth<=12
            and 40<=a.max_plies<=1024 and a.opening_plies<a.max_plies):
        p.error("Invalid arena settings")
    run(a)
if __name__=="__main__":
    main()
