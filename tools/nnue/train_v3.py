#!/usr/bin/env python3
"""Disk-backed, resumable CSR SVNNUE3 Lite trainer.

Native per-perspective king-bucket sparse features -> shared 128 FT -> 32 ->
16 -> signed STM score. Strict heldout game splits and full Adam resumption.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import time

for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(var,"1")

import numpy as np
from dataset_v3 import build,open_db,batches,sparse_batch
from sparse_ft import touched_gradient, update_touched_adagrad
from model_v3 import (initialize,activations,float_predict,integer_predict,
                       quantize,write_network,read_network,digest)

VERSION="svnnue-v3-lite-adam1"
SPARSE_VERSION="svnnue-v3-lite-adam-tail-sparse-ft-adagrad1"
OPTIMIZERS=("dense_adam","sparse_ft_adagrad")

def version_for(optimizer):
    if optimizer not in OPTIMIZERS:
        raise ValueError("Invalid NNUE3 optimizer")
    return VERSION if optimizer=="dense_adam" else SPARSE_VERSION
KEYS=("ft","bft","w1","b1","w2","b2","w3","b3")
RATES={"ft":.002,"bft":.002,"w1":.004,"b1":.004,
       "w2":.005,"b2":.005,"w3":.075,"b3":.3}
REG={"ft":1e-7,"w1":1e-6,"w2":1e-6,"w3":1e-7}
BOUNDS={"ft":8,"bft":8,"w1":8,"b1":8,"w2":8,"b2":8,"w3":255,"b3":10000}

def atomic_json(path,obj):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n")
    tmp.replace(path)

def state_init(desc,db,seed,optimizer="dense_adam"):
    average=db.execute("SELECT AVG(label) FROM examples WHERE part=0").fetchone()[0]
    w=initialize(desc,seed,offset_cp=float(average or 0.))
    return {"weights":w,"moment":{k:np.zeros_like(v) for k,v in w.items()},
            "variance":{k:np.zeros_like(v) for k,v in w.items()},
            "best":{k:v.copy() for k,v in w.items()},
            "epoch":0,"steps":0,"best_epoch":0,"best_loss":float("inf"),
            "history":[],"optimizer":optimizer}

def store(out,state,signature,seed,batch_size):
    epoch=state["epoch"]
    name=f"state-epoch-{epoch:04d}.npz"
    target=out/name
    tmp=out/(name+".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**{f"{prefix}_{k}":v for prefix,arr in
            (("weights",state["weights"]),("moment",state["moment"]),
             ("variance",state["variance"]),("best",state["best"]))
             for k,v in arr.items()})
    tmp.replace(target)
    atomic_json(out/"latest.json",{
        "schema":3,"version":version_for(state["optimizer"]),
        "optimizer":state["optimizer"],"sha256":digest(target),
        "filename":name,"dataset_signature":signature,"seed":seed,
        "batch_size":batch_size,"epoch":epoch,"steps":state["steps"],
        "best_epoch":state["best_epoch"],
        "best_loss":state["best_loss"] if math.isfinite(state["best_loss"]) else None,
        "history":state["history"]})
    for old in out.glob("state-epoch-*.npz"):
        if old.name!=name and old.name!=f"state-epoch-{epoch-1:04d}.npz":
            old.unlink()

def restore(out,signature,seed,batch_size,descriptor,optimizer="dense_adam"):
    report=json.loads((out/"latest.json").read_text())
    if (report.get("schema")!=3 or report.get("version")!=version_for(optimizer)
        or report.get("optimizer", "dense_adam")!=optimizer
        or report.get("dataset_signature")!=signature
        or report.get("batch_size")!=batch_size
        or report.get("seed")!=seed):
        raise ValueError("Refusing v3 optimizer restart with a different dataset/seed/batch")
    filename=report["filename"]
    if not isinstance(filename,str) or Path(filename).name!=filename:
        raise ValueError("Unsafe checkpoint filename")
    checkpoint=out/filename
    if digest(checkpoint)!=report["sha256"]:
        raise ValueError("v3 checkpoint corruption")
    init=initialize(descriptor,seed)
    state={}
    with np.load(checkpoint,allow_pickle=False) as record:
        for prefix in ("weights","moment","variance","best"):
            state[prefix]={}
            for k,example in init.items():
                data=record[f"{prefix}_{k}"]
                if (data.dtype!=np.float32 or data.shape!=example.shape
                    or not np.isfinite(data).all()):
                    raise ValueError("Invalid v3 optimizer parameter shape/nonfinite")
                state[prefix][k]=data.copy()
    state.update({"epoch":report["epoch"],"steps":report["steps"],
                  "best_epoch":report["best_epoch"],
                  "best_loss":report["best_loss"] if report["best_loss"] is not None else float("inf"),
                  "history":report["history"],"optimizer":optimizer})
    return state

def evaluate(w,db,part,dim,batch_size,integer=None):
    total=0
    absolute=squared=0.
    quant=[]
    for records in batches(db,part,batch_size):
        white,black,side,target=sparse_batch(records,dim)
        floats=float_predict(w,white,black,side)
        diff=floats.astype(np.float64)-target.astype(np.float64)
        total+=len(target)
        absolute+=float(np.sum(np.abs(diff)))
        squared+=float(np.sum(diff*diff))
        if integer is not None:
            exact=integer_predict(integer,white,black,side)
            quant.extend(np.abs(exact.astype(np.float64)-floats).tolist())
    if not total:raise ValueError("Missing native heldout games")
    result={"rows":total,"mae_cp":absolute/total,"mse_cp2":squared/total,
            "rmse_cp":math.sqrt(squared/total)}
    if integer is None:return result
    err=np.asarray(quant,dtype=np.float64)
    return result,{"mae_cp":float(err.mean()),"p99_cp":float(np.quantile(err,.99)),
                   "max_cp":float(err.max(initial=0))}

def optimizer_step(state,white,black,side,target,epoch):
    w=state["weights"]
    aw,ab,cat,z1,h1,z2,h2=activations(w,white,black,side)
    predict=(h2@w["w3"]+w["b3"][0]).astype(np.float32)
    scale=2.0/(len(target)*600.0**2)
    delta=(predict-target)*scale
    grads={}
    grads["w3"]=np.asarray(h2.T@delta,dtype=np.float32)
    grads["b3"]=np.asarray([delta.sum()],dtype=np.float32)
    gz2=(delta[:,None]*w["w3"][None,:])*((z2>0)&(z2<1))
    grads["w2"]=np.asarray(h1.T@gz2,dtype=np.float32)
    grads["b2"]=gz2.sum(axis=0).astype(np.float32)
    gz1=(gz2@w["w2"].T)*((z1>0)&(z1<1))
    grads["w1"]=np.asarray(cat.T@gz1,dtype=np.float32)
    grads["b1"]=gz1.sum(axis=0).astype(np.float32)
    gcat=gz1@w["w1"].T
    prew=white@w["ft"]+w["bft"]
    preb=black@w["ft"]+w["bft"]
    gwhite=np.where(side[:,None]==0,gcat[:,:128],gcat[:,128:])
    gblack=np.where(side[:,None]==0,gcat[:,128:],gcat[:,:128])
    gwhite*=((prew>0)&(prew<1))
    gblack*=((preb>0)&(preb<1))
    optimizer=state.get("optimizer","dense_adam")
    if optimizer=="sparse_ft_adagrad":
        active,ft_gradient=touched_gradient(
            white,black,gwhite,gblack,w["ft"].shape[0])
    else:
        grads["ft"]=np.asarray(white.T@gwhite+black.T@gblack,dtype=np.float32)
    grads["bft"]=(gwhite+gblack).sum(axis=0).astype(np.float32)
    state["steps"]+=1
    step=state["steps"]
    schedule=max(.2,math.exp(-epoch/80))
    if optimizer=="sparse_ft_adagrad":
        # Sparse FT optimizer has no momentum for missing rows. In particular,
        # no dense input_dim-by-128 gradient or Adam update is allocated.
        update_touched_adagrad(
            w["ft"],state["variance"]["ft"],active,ft_gradient,
            RATES["ft"]*schedule,REG["ft"],BOUNDS["ft"])
    for key in KEYS:
        if key=="ft" and optimizer=="sparse_ft_adagrad":
            continue
        g=grads[key]
        if key in REG:g+=REG[key]*w[key]
        if not np.isfinite(g).all():raise ValueError(f"Nonfinite Adam gradient: {key}")
        state["moment"][key]=.9*state["moment"][key]+.1*g
        state["variance"][key]=.999*state["variance"][key]+.001*g*g
        corrected=state["moment"][key]/(1-.9**step)
        denominator=np.sqrt(state["variance"][key]/(1-.999**step))+1e-8
        w[key]-=RATES[key]*schedule*corrected/denominator
        np.clip(w[key],-BOUNDS[key],BOUNDS[key],out=w[key])

def final(out,state,summary,db,batch,seed,quant_qa=256):
    if state["best_epoch"]==0:raise ValueError("No validation-selected v3 weights")
    best=state["best"]
    # Network precision lives in the on-disk header. The original native
    # royal-feature schema and training split remain completely unchanged.
    d=dict(summary["descriptor"],qa=quant_qa)
    quant=quantize(best,d)
    model=out/(d["variant"]+".svnnue")
    write_network(model,d,quant)
    loaded=read_network(model,d)
    for k in KEYS:
        if not np.array_equal(quant[k],loaded[k]):
            raise ValueError("Corrupt quantized v3 roundtrip")
    val,qv=evaluate(best,db,1,d["input_dim"],batch,loaded)
    test,qt=evaluate(best,db,2,d["input_dim"],batch,loaded)
    limits={"mae_cp":6.,"p99_cp":35.,"max_cp":100.}
    ok=all(r[k]<=limits[k] for r in (qv,qt) for k in limits)
    with (out/"checkpoint_best.npz").open("wb") as handle:
        np.savez_compressed(handle,**best)
    report={"schema":3,"trainer":version_for(state["optimizer"]),
            "optimizer":state["optimizer"],"descriptor":d,
            "quant_precision_qa":quant_qa,"feature_schema_qa":summary["descriptor"]["qa"],
            "dataset_signature":summary["signature"],
            "seed":seed,"epochs_completed":state["epoch"],"gradient_steps":state["steps"],
            "selected_epoch":state["best_epoch"],"history":state["history"],
            "train_count":summary["partitions"]["train"],"validation":val,"test":test,
            "quantization": {"validation":qv,"test":qt,"limits":limits},
            "quantization_passed":ok,"elo":None,"strength_validated":False}
    atomic_json(out/"metrics.json",report)
    atomic_json(out/"network-manifest.json",{
        "schema":3,"format":"SVNNUE3-Lite",
        "optimizer":state["optimizer"],
        "network":model.name,"network_sha256":digest(model),"descriptor":d,
        "dataset_signature":summary["signature"],
        "metrics_sha256":digest(out/"metrics.json"),
        "optimizer_state_sha256":json.loads((out/"latest.json").read_text())["sha256"],
        "source_manifest_sha256":summary["manifest_sha256"],
        "quantization_passed":ok,"experimental":True,"elo":None})
    if not ok:raise ValueError("v3 quantization error too large")
    return {"model":str(model),"epochs":state["epoch"],"selected_epoch":state["best_epoch"],
            "steps":state["steps"],"heldout_teacher_mae_cp":test["mae_cp"],
            "quantization_passed":ok}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epochs",type=int,default=8)
    p.add_argument("--stop-after-epoch",type=int,default=0)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--seed",type=int,default=20261027)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--optimizer",choices=OPTIMIZERS,default="dense_adam")
    p.add_argument("--quant-qa",type=int,choices=(256,512,1024,2048,4096),default=256)
    a=p.parse_args()
    if not (1<=a.epochs<=1000 and 0<=a.stop_after_epoch<=a.epochs and
            8<=a.batch_size<=512 and 0<=a.seed<2**32):
        p.error("Out-of-range bounded training budget")
    if not a.resume and a.output.exists() and any(a.output.iterdir()):
        p.error("Refusing overwrite of unresumed NNUE experiment")
    a.output.mkdir(parents=True,exist_ok=True)
    summary=build(a.data,a.output/"index.sqlite3",a.seed)
    atomic_json(a.output/"index-summary.json",summary)
    with open_db(a.output/"index.sqlite3") as db:
        state=(restore(a.output,summary["signature"],a.seed,a.batch_size,
                       summary["descriptor"],a.optimizer) if a.resume else
               state_init(summary["descriptor"],db,a.seed,a.optimizer))
        until=a.stop_after_epoch or a.epochs
        if until<=state["epoch"]:raise ValueError("No more epochs to resume")
        for epoch in range(state["epoch"]+1,until+1):
            count=0
            for records in batches(db,0,a.batch_size,a.seed+epoch):
                xw,xb,side,target=sparse_batch(records,summary["descriptor"]["input_dim"])
                optimizer_step(state,xw,xb,side,target,epoch)
                count+=len(target)
            state["epoch"]=epoch
            validation=evaluate(state["weights"],db,1,summary["descriptor"]["input_dim"],a.batch_size)
            state["history"].append({"epoch":epoch,"steps":state["steps"],"train_rows":count,
                                     "validation_mae_cp":validation["mae_cp"],
                                     "validation_mse_cp2":validation["mse_cp2"]})
            if validation["mse_cp2"]<state["best_loss"]:
                state["best_loss"]=validation["mse_cp2"]
                state["best_epoch"]=epoch
                state["best"]={k:v.copy() for k,v in state["weights"].items()}
            store(a.output,state,summary["signature"],a.seed,a.batch_size)
            print(json.dumps({"epoch":epoch,"steps":state["steps"],
                              "validation_mae_cp":round(validation["mae_cp"],3),
                              "best_epoch":state["best_epoch"]}),flush=True)
        if until==a.epochs:
            print(json.dumps(final(a.output,state,summary,db,a.batch_size,a.seed,a.quant_qa)),flush=True)
        else:
            print(json.dumps({"checkpoint_only":True,"epoch":state["epoch"],
                              "source_rows":summary["input_rows"]}),flush=True)

if __name__=="__main__":main()
