#!/usr/bin/env python3
"""Streaming SVNNUE-v1 CPU trainer with integrity-bound full Adam resume.

Indexes millions of native sparse positions on disk; each optimizer call reads
only a bounded mini-batch. Holds test labels out until training completes.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"1")

import numpy as np
from model import HIDDEN, digest, float_predict, integer_predict, quantize, read_network, write_network
from stream_dataset import build_index, batches, dense_batch, open_db, PARTITIONS

VERSION = "svnnue-stream-adam-v1"
RATES = {"w1": .008, "b1": .008, "w2": 1.5, "b2": 1.5}
QUANT_LIMITS = {"mae_cp": 3., "p99_cp": 20., "max_cp": 40.}
KEYS = ("w1","b1","w2","b2")


def json_save(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(target)


def weight_sha(w):
    hasher = hashlib.sha256()
    for key in KEYS:
        hasher.update(key.encode())
        hasher.update(np.asarray(w[key],dtype="<f4").tobytes())
    return hasher.hexdigest()


def initial_state(desc, seed, db):
    rng = np.random.default_rng(seed)
    weights = {
        "w1": rng.normal(0, .025, (desc["input_dim"],HIDDEN)).astype(np.float32),
        "b1": np.full(HIDDEN,.25,dtype=np.float32),
        "w2": rng.normal(0, 5, (2,HIDDEN)).astype(np.float32),
        "b2": np.zeros(2,dtype=np.float32),
    }
    # Per-head centering uses *only* the training partition.
    for color in (0,1):
        row=db.execute("SELECT AVG(label) FROM examples WHERE partition=0 AND side=?",(color,)).fetchone()
        if row[0] is not None: weights["b2"][color]=float(row[0])
    moment={key:np.zeros_like(w) for key,w in weights.items()}
    variance={key:np.zeros_like(w) for key,w in weights.items()}
    best={key:w.copy() for key,w in weights.items()}
    return dict(weights=weights,moment=moment,variance=variance,best=best,
                epoch=0,steps=0,best_loss=float("inf"),best_epoch=0,history=[],
                initial_sha=weight_sha(weights))


def save_state(directory, state, signature, seed, batch_size):
    directory=Path(directory)
    epoch=state["epoch"]
    name=f"state-epoch-{epoch:05d}.npz"
    path=directory/name
    tmp=directory/(name+".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**{
            f"{prefix}_{key}": value[key]
            for prefix,value in (("weights",state["weights"]),("moment",state["moment"]),
                                 ("variance",state["variance"]),("best",state["best"]))
            for key in KEYS})
    tmp.replace(path)
    metadata={"schema":1,"version":VERSION,"file":name,"sha256":digest(path),
              "signature":signature,"seed":seed,"batch_size":batch_size,
              "epoch":epoch,"steps":state["steps"],
              "best_loss":state["best_loss"] if math.isfinite(state["best_loss"]) else None,
              "best_epoch":state["best_epoch"],"history":state["history"],
              "initial_sha":state["initial_sha"]}
    json_save(directory/"latest.json",metadata)
    # Keep the current and preceding checkpoints. An interrupted replace never
    # silently turns into a newly initialized optimizer.
    for old in directory.glob("state-epoch-*.npz"):
        if old.name != name and old.name != f"state-epoch-{epoch-1:05d}.npz":
            old.unlink()


def restore_state(directory, signature, seed, batch_size, desc):
    directory=Path(directory)
    metadata=json.loads((directory/"latest.json").read_text(encoding="utf-8"))
    if (metadata.get("schema")!=1 or metadata.get("version")!=VERSION or
        metadata.get("signature")!=signature or metadata.get("seed")!=seed or
        metadata.get("batch_size")!=batch_size):
        raise ValueError("Refusing checkpoint resume with changed dataset, seed or hyperparameters")
    file=metadata.get("file")
    if not isinstance(file,str) or Path(file).name!=file:
        raise ValueError("Unsafe checkpoint name")
    path=directory/file
    if not path.exists() or digest(path)!=metadata.get("sha256"):
        raise ValueError("Checkpoint checksum mismatch; refusing partial resume")
    expected={"w1":(desc["input_dim"],HIDDEN),"b1":(HIDDEN,),"w2":(2,HIDDEN),"b2":(2,)}
    with np.load(path,allow_pickle=False) as saved:
        state={}
        for prefix in ("weights","moment","variance","best"):
            state[prefix]={}
            for key in KEYS:
                arr=np.asarray(saved[f"{prefix}_{key}"])
                if arr.shape!=expected[key] or arr.dtype!=np.float32 or not np.isfinite(arr).all():
                    raise ValueError("Checkpoint contains an incompatible or non-finite weight array")
                state[prefix][key]=arr.copy()
    state.update(epoch=metadata["epoch"],steps=metadata["steps"],
                 best_loss=metadata["best_loss"] if metadata["best_loss"] is not None else float("inf"),
                 best_epoch=metadata["best_epoch"],history=metadata["history"],
                 initial_sha=metadata["initial_sha"])
    if not isinstance(state["epoch"],int) or not isinstance(state["steps"],int) or state["epoch"]<0 or state["steps"]<0:
        raise ValueError("Invalid checkpoint progress")
    return state


def evaluate(weights, db, part, dim, batch_size, collect_quantized=None):
    sum_abs=sum_sq=total=0
    quant_error=[]
    quant_sq=quant_abs=0.
    for batch in batches(db,part,batch_size,epoch_seed=0):
        x,side,y=dense_batch(batch,dim)
        floating=float_predict(weights,x,side).astype(np.float64)
        diff=floating-y
        sum_abs+=float(np.abs(diff).sum())
        sum_sq+=float((diff*diff).sum())
        total+=len(y)
        if collect_quantized is not None:
            integer=integer_predict(collect_quantized,x.astype(np.int64),side).astype(np.float64)
            change=np.abs(integer-floating)
            quant_error.extend(change.tolist())
            quant_sq+=float(np.sum((integer-y)**2))
            quant_abs+=float(np.sum(np.abs(integer-y)))
    if not total:
        raise ValueError("Empty evaluation partition")
    report={"positions":total,"mae_cp":sum_abs/total,"mse_cp2":sum_sq/total,
            "rmse_cp":math.sqrt(sum_sq/total)}
    if collect_quantized is None:
        return report
    errors=np.asarray(quant_error,dtype=np.float64)
    quant={"mae_cp":float(errors.mean()),"p99_cp":float(np.quantile(errors,.99)),
           "max_cp":float(errors.max(initial=0))}
    return report, quant, {"mae_cp":quant_abs/total,"mse_cp2":quant_sq/total,
                           "rmse_cp":math.sqrt(quant_sq/total),"positions":total}


def train_epoch(state, db, desc, batch_size, seed):
    epoch=state["epoch"]+1
    w=state["weights"]
    moment=state["moment"]
    variance=state["variance"]
    count=0
    for batch in batches(db,0,batch_size,epoch_seed=seed+epoch):
        x, side, y=dense_batch(batch,desc["input_dim"])
        pre=x @ w["w1"] + w["b1"]
        activation=np.clip(pre,0,1)
        pred=np.sum(activation*w["w2"][side],axis=1)+w["b2"][side]
        grad=(pred-y) * (2.0/(len(y)*600.0**2))
        hidden=grad[:,None]*w["w2"][side]*((pre>0)&(pre<1))
        g={"w1":x.T@hidden+1e-6*w["w1"],"b1":hidden.sum(axis=0),
           "w2":np.zeros_like(w["w2"]),"b2":np.zeros_like(w["b2"])}
        for color in (0,1):
            mask=side==color
            if np.any(mask):
                g["w2"][color]=np.sum(activation[mask]*grad[mask,None],axis=0)
                g["b2"][color]=grad[mask].sum()
        state["steps"]+=1
        step=state["steps"]
        # LR depends on completed epoch, not future target epochs. Thus
        # 1 epoch + resume to 3 equals uninterrupted 3 epochs.
        decay=max(.25,math.exp(-epoch/50))
        for key in KEYS:
            if not np.isfinite(g[key]).all():
                raise ValueError("Nonfinite optimizer gradient")
            moment[key]=.9*moment[key]+.1*g[key]
            variance[key]=.999*variance[key]+.001*g[key]*g[key]
            corrected=moment[key]/(1-.9**step)
            denominator=np.sqrt(variance[key]/(1-.999**step))+1e-8
            w[key]-=RATES[key]*decay*corrected/denominator
        for key,limit in (("w1",8),("b1",8),("w2",500),("b2",10000)):
            np.clip(w[key],-limit,limit,out=w[key])
        count+=len(y)
    if count==0:
        raise ValueError("No examples for optimizer")
    state["epoch"]=epoch
    return count


def final_export(state, out, summary, db, batch_size, seed):
    weights=state["best"]
    desc=summary["descriptor"]
    if state["steps"]==0 or state["best_epoch"]<1:
        raise ValueError("No trained checkpoint to export")
    quant=quantize(weights,desc)
    network=out/(summary["variant"]+".svnnue")
    write_network(network,desc,quant)
    reloaded=read_network(network,desc)
    if any(not np.array_equal(quant[key],reloaded[key]) for key in KEYS):
        raise ValueError("Quantized serialization mismatch")
    reports={}
    quant_reports={}
    quant_errors={}
    for part in (1,2):
        label=PARTITIONS[part]
        reports[label],quant_errors[label],quant_reports[label]=evaluate(
            weights,db,part,desc["input_dim"],batch_size,reloaded)
    gate=all(stat[k]<=QUANT_LIMITS[k] for stat in quant_errors.values() for k in QUANT_LIMITS)
    np.savez_compressed(out/"checkpoint.npz",**weights,selected_epoch=np.asarray(state["best_epoch"]))
    record={
        "schema":1,"trainer_version":VERSION,"descriptor":desc,"stream_index_signature":summary["signature"],
        "source_manifests":summary["source_manifests"],"teacher":summary["teacher"],
        "seed":seed,"epochs_completed":state["epoch"],"gradient_steps":state["steps"],
        "selected_epoch":state["best_epoch"],"initial_weights_sha256":state["initial_sha"],
        "trained_weights_sha256":weight_sha(weights),"history":state["history"],
        "test_teacher_approximation":reports["test"],"validation_teacher_approximation":reports["validation"],
        "quantization_error":quant_errors,"quantized":quant_reports,
        "quantization_passed":gate,"quantization_limits":QUANT_LIMITS,
        "rows":summary["partition_rows"],"duplicates_removed":summary["duplicates_removed"],
        "evaluation":"classical-teacher-distillation","strength_validated":False,"elo":None,
        "training_completed":True}
    json_save(out/"metrics.json",record)
    model_manifest={"schema":1,"format":"SVNNUE-v1","descriptor":desc,"network":network.name,
         "network_sha256":digest(network),"network_bytes":network.stat().st_size,
         "checkpoint_sha256":digest(out/"checkpoint.npz"),"metrics_sha256":digest(out/"metrics.json"),
         "stream_index_signature":summary["signature"],"trainer_version":VERSION,
         "license":"AGPL-3.0-only","default_enabled":False,"strength_validated":False,
         "status":"experimental","quantization_passed":gate}
    json_save(out/"network-manifest.json",model_manifest)
    if not gate:
        raise ValueError("Quantization error exceeded contract; model must not be promoted")
    return {"network":str(network),"epoch":state["epoch"],
            "selected_epoch":state["best_epoch"],"steps":state["steps"],
            "test_mae_cp":reports["test"]["mae_cp"],"quantization_passed":gate}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",nargs="+",required=True,type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--index",type=Path,default=None)
    p.add_argument("--epochs",type=int,default=10)
    p.add_argument("--stop-after-epoch",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--seed",type=int,default=20261026)
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if (not 0<args.epochs<=5000 or not 8<=args.batch_size<=1024
        or not 0<=args.seed<2**32 or not 0<=args.stop_after_epoch<=args.epochs):
        p.error("Invalid epochs/batch/seed/stop limit")
    out=args.output.resolve()
    index=args.index.resolve() if args.index else out/"index.sqlite3"
    if not args.resume and out.exists() and any(out.iterdir()):
        p.error("Output is not empty; use --resume only with the same dataset")
    out.mkdir(parents=True,exist_ok=True)
    summary=build_index(args.data,index,args.seed)
    json_save(out/"index-summary.json",summary)
    db=open_db(index)
    started=time.monotonic()
    try:
        state=restore_state(out,summary["signature"],args.seed,args.batch_size,
                            summary["descriptor"]) if args.resume else initial_state(
                                summary["descriptor"],args.seed,db)
        if state["epoch"]>=args.epochs:
            raise ValueError("Checkpoint already reached requested total epochs")
        target=args.stop_after_epoch or args.epochs
        if target<=state["epoch"]:
            raise ValueError("Requested stop epoch is not ahead of current checkpoint")
        for epoch in range(state["epoch"]+1,target+1):
            visited=train_epoch(state,db,summary["descriptor"],args.batch_size,args.seed)
            metrics=evaluate(state["weights"],db,1,summary["descriptor"]["input_dim"],args.batch_size)
            state["history"].append({"epoch":epoch,"validation":metrics,"train_rows_seen":visited,
                                     "steps":state["steps"]})
            if metrics["mse_cp2"] < state["best_loss"]:
                state["best_loss"]=metrics["mse_cp2"]
                state["best_epoch"]=epoch
                state["best"]={k:state["weights"][k].copy() for k in KEYS}
            save_state(out,state,summary["signature"],args.seed,args.batch_size)
            print(json.dumps({"epoch":epoch,"steps":state["steps"],
                              "validation_mae_cp":round(metrics["mae_cp"],3),
                              "best_epoch":state["best_epoch"]}),flush=True)
        if target!=args.epochs:
            print(json.dumps({"checkpoint_only":True,"epoch":state["epoch"],
                              "index_rows":summary["rows_indexed"]}))
        else:
            print(json.dumps(final_export(state,out,summary,db,args.batch_size,args.seed)),flush=True)
    finally:
        db.close()
    print(json.dumps({"elapsed_seconds":round(time.monotonic()-started,2),
                      "index_rows":summary["rows_indexed"],
                      "peak_dense_examples_per_batch":args.batch_size}),flush=True)


if __name__=="__main__":
    main()
