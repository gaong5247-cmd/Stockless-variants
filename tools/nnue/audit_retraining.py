#!/usr/bin/env python3
"""Audit original SVNNUE3 data and actual NEW optimizer gradient steps.

No new generation is performed. Refuses altered 8,153,309-source corpora,
untested model rebindings, bad quantization and mismatched native predictions.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_source(proof, merged):
    p=load(proof)
    m=load(merged)
    if (m["variant"]!=p["variant"] or m["games"]!=p["games"] or
        m["games"]!=12288 or m["positions"]!=p["generated_positions"] or
        m["descriptor"]!=p["descriptor"] or
        m["native_engine_sha256"]!=p["native_teacher_sha256"] or
        len(m["source_shards"])!=12 or not p["unique_target_met"] or
        p["unique_model_inputs_sha256"]<1000000):
        raise ValueError("Existing original million-position source certification disagrees")
    for actual, source in zip(m["source_shards"],p["source_shards"]):
        for x,y in (("shard","id"),("manifest_sha256","manifest_sha256"),
                    ("positions","positions"),("seed","seed")):
            if actual[x]!=source[y]:
                raise ValueError("Original SHA-bound shard metadata mismatch")
    return p,m


def verify_model(proof,merged,model_dir):
    p,m=verify_source(proof,merged)
    root=Path(model_dir)
    idx=load(root/"index-summary.json")
    metric=load(root/"metrics.json")
    native=load(root/"native-validation.json")
    att=load(root/"feature-compatibility.json")
    net=load(root/"network-manifest.json")
    if (idx["input_rows"]!=m["positions"] or
        metric["epochs_completed"]!=6 or metric["gradient_steps"]<=0 or
        metric["optimizer"]!="sparse_ft_adagrad" or
        not metric["quantization_passed"] or
        metric["quant_precision_qa"]!=2048 or
        metric["native_rebind_attested"] is not True or
        net["descriptor"]["qa"]!=2048 or
        net["descriptor"]["rules_sha256"]!=att["target_rules_sha256"] or
        net["dataset_signature"]!=idx["signature"] or
        native["failed"]!=0 or native["passed"]!=132 or
        att["source_labels_modified"] is not False or
        att["source_dataset_signature"]!=idx["signature"]):
        raise ValueError("Actual newly optimized weights did not pass original-data native gate")
    return {"variant":p["variant"],"original_games":m["games"],
            "original_positions":m["positions"],
            "certified_distinct_inputs":p["unique_model_inputs_sha256"],
            "new_gradient_steps":metric["gradient_steps"],
            "training_epochs":metric["epochs_completed"],
            "best_epoch":metric["selected_epoch"],
            "teacher_test_mae_cp":metric["test"]["mae_cp"],
            "quantization_qa":net["descriptor"]["qa"],
            "native_checks":native["passed"],
            "generation_jobs_started":0,
            "elo":None,"strength_validated":False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--proof",required=True,type=Path)
    parser.add_argument("--merged",required=True,type=Path)
    parser.add_argument("--model-dir",type=Path)
    args=parser.parse_args()
    if args.model_dir:
        result=verify_model(args.proof,args.merged,args.model_dir)
    else:
        p,m=verify_source(args.proof,args.merged)
        result={"variant":p["variant"],"original_rows":m["positions"],
                "certified_distinct_inputs":p["unique_model_inputs_sha256"],
                "sha_bound_original_shards":len(m["source_shards"]),
                "no_new_data":True}
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
