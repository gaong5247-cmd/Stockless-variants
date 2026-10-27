#!/usr/bin/env python3
"""Allow reusing old SOURCE NNUE3 data only after native feature parity.

Old teacher identity and new C++ source/rule fingerprint necessarily differ
when inference precision changes. This command independently replays frozen
heldout *full game histories* in both native engines, and refuses mismatched
FENs, legal results, king buckets, sparse features or source identity.
It does NOT re-label old teacher positions or claim full rule equivalence.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from dataset_v3 import build, batches, open_db, unpack
from generate import Engine, sha256

PREFIX="info string NNUE_FEATURES "
def state(engine,moves):
    engine.position(moves)
    engine.send("nnuefeaturesv3")
    msg=engine.until(lambda line:line.startswith(PREFIX))[-1]
    data=json.loads(msg[len(PREFIX):])
    if "v3" not in data or "error" in data["v3"]:
        raise ValueError("Native v3 feature export failed")
    return data

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--old-engine",required=True,type=Path)
    p.add_argument("--new-engine",required=True,type=Path)
    p.add_argument("--seed",type=int,default=20261027)
    p.add_argument("--positions",type=int,default=128)
    a=p.parse_args()
    if not 64<=a.positions<=1024:p.error("Need 64-1024 heldout positions")
    a.out.mkdir(parents=True,exist_ok=True)
    summary=build(a.source,a.out/"index.sqlite3",a.seed)
    (a.out/"index-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    old_sha=sha256(a.old_engine)
    new_sha=sha256(a.new_engine)
    if old_sha!=summary["teacher_engine_sha256"]:
        raise ValueError("Old native executable does not match the original 8M data producer")
    if old_sha==new_sha:
        raise ValueError("Expected separately built old/new binaries")
    selection=[]
    moves={}
    with open_db(a.out/"index.sqlite3") as db:
        signature=db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if signature is None or signature[0]!=summary["signature"]:
            raise ValueError("Native source SQLite index identity missing")
        for batch in batches(db,2,64,epoch_seed=a.seed):
            for item in batch:
                selection.append(item)
                if len(selection)>=a.positions:break
            if len(selection)>=a.positions:break
        if len(selection)<a.positions:
            raise ValueError("Fewer than requested independent heldout positions")
        for item in selection:
            gid=item[5]
            if gid not in moves:
                rows=db.execute("SELECT moves FROM games WHERE game_id=?",(gid,)).fetchone()
                if not rows:raise ValueError("Original game moves missing")
                moves[gid]=json.loads(rows[0])
    engines=[Engine(a.old_engine.resolve(),90),Engine(a.new_engine.resolve(),90)]
    try:
        variant=summary["descriptor"]["variant"]
        for engine in engines:
            engine.option("UCI_Variant",variant)
            engine.option("Threads",1)
            engine.option("Hash",16)
            engine.option("StocklessHybrid",True)
            engine.option("Use NNUE",False)
            engine.option("EvalBackend","Classical")
        actual_rules=None
        for item in selection:
            _row,white,black,stm,_label,gid,ply,fen=item
            game=moves[gid][:ply]
            state_old=state(engines[0],game)
            state_new=state(engines[1],game)
            original=state_old["v3"]
            current=state_new["v3"]
            if original["descriptor"]!=summary["descriptor"]:
                raise ValueError("Frozen producer description mismatches its original dataset")
            old_core={k:v for k,v in original["descriptor"].items() if k!="rules_sha256"}
            new_core={k:v for k,v in current["descriptor"].items() if k!="rules_sha256"}
            if old_core!=new_core:
                raise ValueError("The new C++ royal-index input schema is incompatible")
            if actual_rules is None:
                actual_rules=current["descriptor"]["rules_sha256"]
            elif actual_rules!=current["descriptor"]["rules_sha256"]:
                raise ValueError("New native rule fingerprint changed during validation")
            if (state_old["fen"]!=fen or state_new["fen"]!=fen
                or state_old["side_to_move"]!=stm or state_new["side_to_move"]!=stm
                or state_old["terminal"]!=state_new["terminal"]
                or state_old["legal_moves"]!=state_new["legal_moves"]):
                raise ValueError("Native game state, result, turn or legal moves diverged")
            for key,expected in (("features_white",white),("features_black",black)):
                target=[[int(i),int(value)] for i,value in unpack(expected)]
                if original[key]!=target or current[key]!=target:
                    raise ValueError("A native royal sparse feature differed in old/new C++ build")
            if (original["white_bucket"]!=current["white_bucket"] or
                original["black_bucket"]!=current["black_bucket"]):
                raise ValueError("King-region buckets differ under new native rules")
        if actual_rules==summary["descriptor"]["rules_sha256"]:
            raise ValueError("Expected a conservative new runtime source/rules fingerprint")
        result={
            "schema":3,"contract":"svnnue3-explicit-feature-rebind-v1",
            "variant":variant,"source_dataset_signature":summary["signature"],
            "source_rules_sha256":summary["descriptor"]["rules_sha256"],
            "target_rules_sha256":actual_rules,
            "old_native_engine_sha256":old_sha,
            "new_native_engine_sha256":new_sha,
            "frozen_real_heldout_positions_checked":len(selection),
            "native_features_matched":True,
            "native_game_rules_matched_on_checked_positions":True,
            "full_rules_equivalence_proven":False,
            "source_labels_modified":False,
            "training_strength_certified":False}
        (a.out/"feature-compatibility.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result),flush=True)
    finally:
        for engine in engines:engine.close()

if __name__=="__main__":main()
