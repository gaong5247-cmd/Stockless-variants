#!/usr/bin/env python3
"""Create checksum-verified persistent archives of independently audited v3 data.

Every shard is v2 teacher-labeled historical data RE-ENCODED by the native v3
rules engine. No new teacher searches were performed. Original model files
are strictly tied to their exact native inference version.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

SOURCE_TRAIN_RUN=40333649652
SOURCE_COMMIT="e8a7da1ecf7f815c6369a1e3921a9728d43375a5"
SOURCE_TEACHER_RUN=40329751018
VARIANTS=("chess","shogi","janggi","xiangqi","makruk")
EXPECTED_NETWORKS={
    "chess":"d672db1ae19f8c3a29d49c9fe137050e5de670ca288235060d32c88a2fe4fc48",
    "shogi":"9b52f4cb37c051525468f831c9da9445c16a98811c71954a3d5292ef9b5f75f8",
    "janggi":"0753476ad6483f74bfba5717f04085ac76becd18479968e0b09f17ad86dd083c",
    "xiangqi":"17d89ef1cc0ea9da1ee8973b3e836f6be868e1e1576c462e158ebc0db7a517dc",
    "makruk":"8b987f720b4c1e51acdf459b2c9527245affea55c17778b31d69ff07ecd7b284",
}


def checksum(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as inp:
        for block in iter(lambda:inp.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def verify_bytes(path,expected):
    if not path.is_file() or path.stat().st_size!=expected["bytes"] or checksum(path)!=expected["sha256"]:
        raise ValueError("Source dataset SHA256/size mismatch: "+str(path))


def pack(destination,filename,files,report):
    path=destination/filename
    with ZipFile(path,"w",compression=ZIP_DEFLATED,compresslevel=6,allowZip64=True) as zipfile:
        for name,source in sorted(files.items()):
            zipfile.write(source,name)
        zipfile.writestr("manifest-integrity.json",json.dumps(report,indent=2,sort_keys=True)+"\n")
    with ZipFile(path) as reader:
        if reader.testzip() is not None:
            raise ValueError("Corrupted ZIP archive after writing")
        for name,meta in report["files"].items():
            size=0
            h=hashlib.sha256()
            with reader.open(name) as stream:
                for chunk in iter(lambda:stream.read(1<<20),b""):
                    h.update(chunk)
                    size+=len(chunk)
            if size!=meta["bytes"] or h.hexdigest()!=meta["sha256"]:
                raise ValueError("ZIP member mismatch: "+name)
    (destination/(filename+".sha256")).write_text(checksum(path)+"  "+filename+"\n")
    return {"name":filename,"sha256":checksum(path),"bytes":path.stat().st_size}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--incoming",type=Path,default=Path("incoming"))
    p.add_argument("--output",type=Path,default=Path("release-assets"))
    p.add_argument("--repository",type=Path,default=Path("."))
    args=p.parse_args()
    dest=args.output
    dest.mkdir(parents=True,exist_ok=True)
    overall={"schema":1,"source_run":SOURCE_TRAIN_RUN,"source_commit":SOURCE_COMMIT,
             "original_teacher_run":SOURCE_TEACHER_RUN,"network_format":"SVNNUE-v3",
             "new_teacher_games":0,"new_teacher_positions":0,
             "historical_label_positions_reencoded":0,"unique_native_v3_inputs":0,
             "models":{},"archives":[],
             "note":"Previously generated classical teacher labels; native feature re-encoding, NOT new teacher search.",
             "strength_certified":False,"elo":None}
    source_rules=None
    for v in VARIANTS:
        model_path=args.incoming/v/"model"
        model_manifest=json.loads((model_path/"network-manifest.json").read_text())
        metrics=json.loads((model_path/"metrics.json").read_text())
        index=json.loads((model_path/"index-summary.json").read_text())
        native=json.loads((model_path/"native-validation.json").read_text())
        weights=model_path/(v+".svnnue")
        if (checksum(weights)!=EXPECTED_NETWORKS[v] or
            model_manifest.get("network_sha256")!=EXPECTED_NETWORKS[v] or
            weights.read_bytes()[:8]!=b"SVNNUE3\x00" or
            model_manifest["descriptor"]["variant"]!=v or
            model_manifest["descriptor"]["format_version"]!=3 or
            model_manifest["descriptor"]["feature_version"]!=3 or
            model_manifest["stream_index_signature"]!=index["signature"] or
            metrics["stream_index_signature"]!=index["signature"] or
            metrics.get("epochs_completed")!=24 or
            metrics.get("compute_kernel")!="csr" or
            metrics.get("quantization_passed") is not True or
            native.get("failed")!=0 or native.get("checks")!=132 or
            native.get("network_sha256")!=EXPECTED_NETWORKS[v]):
            raise ValueError("Incomplete v3 trained model/native runtime evidence: "+v)
        if source_rules is None:
            source_rules=index["descriptor"]["rules_sha256"]
        elif source_rules!=index["descriptor"]["rules_sha256"]:
            raise ValueError("Different v3 native rule/feature fingerprints")
        metadata=json.loads((model_path/"latest.json").read_text())
        checkpoint=model_path/metadata["file"]
        if (metadata.get("epoch")!=24 or metadata.get("signature")!=index["signature"]
                or not checkpoint.is_file() or checksum(checkpoint)!=metadata.get("sha256")):
            raise ValueError("Missing or corrupted full Adam continuation state: "+v)
        model_files={f.name:f for f in model_path.iterdir() if f.is_file()}
        if ("latest.json" not in model_files or
            not any(name.startswith("state-epoch-") for name in model_files)):
            raise ValueError("Incomplete model checkpoint package")
        for name in ("LICENSE","NOTICE.md"):
            model_files[name]=args.repository/name
        total=0
        for shard in (0,1):
            path=args.incoming/v/"data"/str(shard)
            source=path/"manifest.json"
            dataset=json.loads(source.read_text())
            if (dataset.get("variant")!=v or dataset.get("game_count")!=1024 or
                dataset.get("relabeling") is not False or
                dataset.get("original_teacher_scores_preserved") is not True or
                dataset.get("source_data_feature_version")!=2 or
                dataset.get("feature_exporter_source_commit")!=SOURCE_COMMIT or
                dataset.get("native_feature_version")!=3 or
                dataset["descriptor"]!=index["descriptor"]):
                raise ValueError("Invalid native v2-to-v3 data provenance: "+v+"/"+str(shard))
            if source_rules!=dataset["rules_sha256"]:
                raise ValueError("Mixed v3 source rule identities")
            files={"manifest.json":source}
            for name in ("games.jsonl","positions.jsonl"):
                current=path/name
                verify_bytes(current,dataset["files"][name])
                expected=dataset["game_count"] if name=="games.jsonl" else dataset["rows"]
                with current.open("rb") as stream:
                    if sum(1 for _ in stream)!=expected:
                        raise ValueError("Source count mismatch")
                files[name]=current
            for name in ("LICENSE","NOTICE.md"):
                files[name]=args.repository/name
            audit={"schema":1,"game":v,"shard":shard,"source_run":SOURCE_TRAIN_RUN,
                   "original_teacher_run":SOURCE_TEACHER_RUN,
                   "label_rows":dataset["rows"],"reencoded":True,"teacher_scores_preserved":True,
                   "files":{name:{"bytes":f.stat().st_size,"sha256":checksum(f)} for name,f in files.items()}}
            result=pack(dest,"svnnue-v3-"+v+"-data-"+str(shard)+".zip",files,audit)
            overall["archives"].append(result)
            total+=dataset["rows"]
        if index.get("rows_read")!=total:
            raise ValueError("Merged dataset row count mismatch "+v)
        audit={"schema":1,"game":v,"source_run":SOURCE_TRAIN_RUN,
               "network_sha256":EXPECTED_NETWORKS[v],
               "optimizer_epoch":metadata["epoch"],"runtime_checks":native["checks"],
               "files":{name:{"bytes":f.stat().st_size,"sha256":checksum(f)} for name,f in model_files.items()}}
        overall["archives"].append(pack(dest,"svnnue-v3-"+v+"-model-and-adam.zip",model_files,audit))
        overall["historical_label_positions_reencoded"]+=total
        overall["unique_native_v3_inputs"]+=index["rows_indexed"]
        overall["models"][v]={"position_rows":total,"unique":index["rows_indexed"],
            "network_sha256":EXPECTED_NETWORKS[v],
            "selected_epoch":metrics["selected_epoch"],
            "runtime_checks":native["checks"],
            "test_mae_cp":metrics["test_teacher_approximation"]["mae_cp"]}
    if (len(overall["archives"])!=15 or overall["historical_label_positions_reencoded"]!=1241675
         or overall["unique_native_v3_inputs"]!=1226999
         or sum(x["runtime_checks"] for x in overall["models"].values())!=660):
        raise ValueError("Missing corpus, models or incorrect total")
    overall["rules_sha256"]=source_rules
    (dest/"EXPERIMENT_MANIFEST.json").write_text(json.dumps(overall,indent=2,sort_keys=True)+"\n")
    (dest/"SHA256SUMS.txt").write_text("".join(
        item["sha256"]+"  "+item["name"]+"\n" for item in overall["archives"]))
    print(json.dumps({"models":len(overall["models"]),
                      "historical_positions":overall["historical_label_positions_reencoded"],
                      "unique_v3_inputs":overall["unique_native_v3_inputs"],
                      "native_checks":660,"archive_count":len(overall["archives"])}))


if __name__=="__main__":
    main()
