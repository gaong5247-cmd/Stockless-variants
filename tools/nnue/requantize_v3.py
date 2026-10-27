#!/usr/bin/env python3
"""Re-quantize already-trained 8M native-position V3 weights, without retraining.

Only the network's header-encoded fixed-point scale changes (QA 256->2048).
The original six-epoch optimizer checkpoint, best float parameters, source
signatures and all original labels remain untouched. A separate native C++
feature/royal/rule replay attestation is required before rebinding the
conservatively changing compiled-rule SHA to the new evaluator binary.

Absolutely no loosening of MAE/p99/max float-vs-integer quantization gates.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
from dataset_v3 import open_db,batches,sparse_batch
from model_v3 import quantize,write_network,read_network,digest,float_predict,integer_predict
from train_v3 import restore,atomic_json,KEYS

LIMITS={"mae_cp":6.0,"p99_cp":35.0,"max_cp":100.0}

def evaluate_quantized(best,quant,db,partition,dim,batch_size):
    deviations=[]
    count=0
    raw_teacher_abs=0.
    native_teacher_abs=0.
    native_squared=0.
    for records in batches(db,partition,batch_size,epoch_seed=0):
        white,black,side,target=sparse_batch(records,dim)
        raw=float_predict(best,white,black,side).astype(np.float64)
        deployed=integer_predict(quant,white,black,side).astype(np.float64)
        y=target.astype(np.float64)
        deviations.extend(np.abs(deployed-raw).tolist())
        count+=len(y)
        raw_teacher_abs+=float(np.abs(raw-y).sum())
        native_teacher_abs+=float(np.abs(deployed-y).sum())
        native_squared+=float(np.sum((deployed-y)**2))
    if not count:
        raise ValueError("Missing native game-safe heldout data")
    values=np.asarray(deviations,dtype=np.float64)
    return {
        "positions":count,
        "quantization":{"mae_cp":float(values.mean()),
                        "p99_cp":float(np.quantile(values,.99)),
                        "max_cp":float(values.max(initial=0))},
        "raw_float_teacher_mae_cp":raw_teacher_abs/count,
        "deployed_native_integer_teacher_mae_cp":native_teacher_abs/count,
        "deployed_native_integer_teacher_mse_cp2":native_squared/count,
        "mae_degradation_cp":(native_teacher_abs-raw_teacher_abs)/count
    }

def run(a):
    checkpoint=a.checkpoint_dir.resolve()
    out=a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite previously exported weights")
    out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((a.index_dir/"index-summary.json").read_text())
    manifest_old=json.loads((checkpoint/"network-manifest.json").read_text())
    source_metrics=json.loads((checkpoint/"metrics.json").read_text())
    native=json.loads((a.index_dir/"feature-compatibility.json").read_text())
    signature=summary["signature"]
    if (manifest_old["dataset_signature"]!=signature
        or source_metrics["dataset_signature"]!=signature
        or manifest_old["network_sha256"]!=digest(
            checkpoint/manifest_old["network"])
        or native.get("contract")!="svnnue3-explicit-feature-rebind-v1"
        or native.get("source_dataset_signature")!=signature
        or native.get("source_rules_sha256")!=summary["descriptor"]["rules_sha256"]
        or native.get("source_labels_modified") is not False
        or native.get("native_features_matched") is not True
        or native.get("frozen_real_heldout_positions_checked",0)<128
        or native.get("target_rules_sha256")==native.get("source_rules_sha256")):
        raise ValueError("Unverifiable old dataset or missing explicit native feature attestation")
    weights=restore(checkpoint,signature,a.seed,a.batch_size,summary["descriptor"],
                    optimizer="sparse_ft_adagrad")
    old_meta=json.loads((checkpoint/"latest.json").read_text())
    if (weights["epoch"]!=6 or source_metrics["epochs_completed"]!=6
        or weights["steps"]!=source_metrics["gradient_steps"]
        or weights["best_epoch"]!=source_metrics["selected_epoch"]
        or source_metrics["optimizer"]!="sparse_ft_adagrad"
        or old_meta["sha256"]!=digest(checkpoint/old_meta["filename"])):
        raise ValueError("Existing 8M training checkpoint did not pass full optimizer audit")
    changed_desc=dict(summary["descriptor"],
                      qa=a.qa,rules_sha256=native["target_rules_sha256"])
    q=quantize(weights["best"],changed_desc)
    path=out/(changed_desc["variant"]+".svnnue")
    write_network(path,changed_desc,q)
    readback=read_network(path,changed_desc)
    for key in KEYS:
        if not np.array_equal(q[key],readback[key]):
            raise ValueError("Quantized v3 network serialization corrupted")
    shutil.copyfile(a.index_dir/"index-summary.json",out/"index-summary.json")
    shutil.copyfile(a.index_dir/"feature-compatibility.json",out/"feature-compatibility.json")
    # Publish complete original optimizer state rather than pretending
    # calibration trained a new model: only the integer export was changed.
    for name in ("latest.json",old_meta["filename"]):
        shutil.copyfile(checkpoint/name,out/name)
    with open_db(a.index_dir/"index.sqlite3") as db:
        sig=db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if not sig or sig[0]!=signature:
            raise ValueError("Source split database has different training signature")
        validation=evaluate_quantized(weights["best"],readback,db,1,
                                      changed_desc["input_dim"],a.batch_size)
        test=evaluate_quantized(weights["best"],readback,db,2,
                                changed_desc["input_dim"],a.batch_size)
    numeric_pass=all(
        report["quantization"][metric]<=limit
        for report in (validation,test)
        for metric,limit in LIMITS.items())
    # Raw float approximation is a historical diagnostic; integer deployment
    # must also not materially worsen *actual* classical-teacher accuracy.
    teacher_quality=all(
        report["mae_degradation_cp"]<=max(3.,report["raw_float_teacher_mae_cp"]*.03)
        for report in (validation,test))
    passed=numeric_pass and teacher_quality
    metrics={
        "schema":3,"run_type":"export_higher_precision_from_existing_trained_checkpoint",
        "optimizer":source_metrics["optimizer"],
        "epochs_completed":weights["epoch"],
        "gradient_steps":weights["steps"],
        "selected_epoch":weights["best_epoch"],
        "trained_on_original_8m_source":True,
        "new_gradient_steps":0,
        "quant_qa":a.qa,"quant_qb":changed_desc["qb"],
        "original_quant_qa":summary["descriptor"]["qa"],
        "source_training_checkpoint_sha256":old_meta["sha256"],
        "dataset_signature":signature,
        "source_rules_sha256":native["source_rules_sha256"],
        "target_runtime_rules_sha256":native["target_rules_sha256"],
        "feature_compatibility_attested":True,
        "network_sha256":digest(path),
        "validation":validation,"test":test,
        "limits_unchanged":LIMITS,
        "quantization_limits_passed":numeric_pass,
        "integer_teacher_fidelity_passed":teacher_quality,
        "quantization_passed":passed,
        "strength_validated":False,"elo":None
    }
    atomic_json(out/"metrics.json",metrics)
    atomic_json(out/"network-manifest.json",{
        "schema":3,"format":"SVNNUE3-Lite",
        "network":path.name,"network_sha256":digest(path),
        "descriptor":changed_desc,
        "dataset_signature":signature,
        "metrics_sha256":digest(out/"metrics.json"),
        "optimizer":"sparse_ft_adagrad",
        "optimizer_checkpoint":old_meta["filename"],
        "optimizer_state_sha256":old_meta["sha256"],
        "source_manifest_sha256":summary["manifest_sha256"],
        "source_model_sha256":manifest_old["network_sha256"],
        "rebind_attestation_sha256":digest(out/"feature-compatibility.json"),
        "quantization_passed":passed,
        "experimental":True,"elo":None
    })
    concise={"variant":changed_desc["variant"],"precision_qa":a.qa,
             "source_training_epochs":weights["epoch"],"native_integer_network":str(path),
             "validation_quantization":validation["quantization"],
             "test_quantization":test["quantization"],
             "deployed_test_teacher_mae_cp":test["deployed_native_integer_teacher_mae_cp"],
             "raw_float_test_teacher_mae_cp":test["raw_float_teacher_mae_cp"],
             "quantization_passed":passed}
    print(json.dumps(concise),flush=True)
    if not passed:
        raise ValueError("V3 signed fixed-point gate STILL fails; reject network without weakening thresholds")
    return concise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir",type=Path,required=True)
    p.add_argument("--index-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--qa",type=int,choices=(512,1024,2048,4096),default=2048)
    p.add_argument("--seed",type=int,default=20261027)
    p.add_argument("--batch-size",type=int,default=256)
    a=p.parse_args()
    run(a)

if __name__=="__main__":main()
