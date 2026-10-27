"""Integrity-bound, disk-backed, dual-perspective SVNNUE3 training samples.

Only the requested minibatch is inflated into CSR tensors in memory. The
original JSONL histories remain authoritative, with whole-game splits.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sqlite3
import struct
import numpy as np
from scipy.sparse import csr_matrix
from model_v3 import digest,validate_descriptor

PACK=struct.Struct("<HH")
PARTS=("train","validation","test")
BUCKETS=32

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)

def split(gid,seed):
    value=hashlib.sha256(f"v3-split:{seed}:{gid}".encode()).digest()[0]%6
    return 1 if value==0 else 2 if value==1 else 0

def encode(pairs,dim):
    prev=-1
    mass=0
    packed=bytearray()
    for pair in pairs:
        if (not isinstance(pair,list) or len(pair)!=2 or
            type(pair[0]) is not int or type(pair[1]) is not int):
            raise ValueError("Bad v3 native sparse feature")
        idx,count=pair
        if not (prev<idx<dim and 0<count<=512):
            raise ValueError("Unsorted, duplicate or invalid v3 feature")
        prev=idx
        mass+=count
        packed.extend(PACK.pack(idx,count))
    if not packed or mass>512:
        raise ValueError("Invalid dual-perspective mass or empty feature set")
    return bytes(packed)

def unpack(blob):
    return list(struct.iter_unpack("<HH",blob))

def open_db(path):
    db=sqlite3.connect(str(path))
    db.execute("PRAGMA cache_size=-4096")
    db.execute("PRAGMA temp_store=FILE")
    return db

def build(data,out,seed):
    data=Path(data)
    manifest=json.loads((data/"manifest.json").read_text())
    if (manifest.get("schema")!=3 or manifest.get("generator") not in
        ("svnnue3-native-selfplay-1","svnnue3-native-selfplay-2")):
        raise ValueError("Invalid native v3 generator version")
    descriptor=manifest["descriptor"]
    dim=validate_descriptor(descriptor)
    for filename in ("games.jsonl","positions.jsonl"):
        src=data/filename
        info=manifest["files"][filename]
        if digest(src)!=info["sha256"] or src.stat().st_size!=info["bytes"]:
            raise ValueError("Dataset checksum/byte length mismatch")
    source=digest(data/"manifest.json")
    signature=hashlib.sha256(canonical({"source":source,"seed":seed,
                                         "descriptor":descriptor}).encode()).hexdigest()
    out=Path(out)
    if out.exists():
        with open_db(out) as db:
            row=db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
            if not row or row[0]!=signature:raise ValueError("Dataset/seed mismatch")
            return json.loads(db.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()[0])
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_name(out.name+".building")
    if tmp.exists():tmp.unlink()
    db=open_db(tmp)
    try:
        db.executescript("""PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;
          CREATE TABLE games(game_id TEXT PRIMARY KEY,part INTEGER NOT NULL,
                  moves TEXT NOT NULL, expected INTEGER NOT NULL,
                  seen INTEGER NOT NULL DEFAULT 0);
          CREATE TABLE examples(id INTEGER PRIMARY KEY,input_key TEXT UNIQUE,
                  part INTEGER NOT NULL,bucket INTEGER NOT NULL,
                  white BLOB NOT NULL,black BLOB NOT NULL,
                  stm INTEGER NOT NULL,label REAL NOT NULL,
                  game_id TEXT NOT NULL,ply INTEGER NOT NULL,fen TEXT NOT NULL);
          CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        """)
        gamecount=0
        with (data/"games.jsonl").open() as handle:
            for line in handle:
                g=json.loads(line)
                if (g.get("schema")!=3 or g.get("variant")!=descriptor["variant"]
                    or not isinstance(g.get("moves"),list) or len(g["moves"])!=g.get("plies")
                    or type(g.get("rows")) is not int):
                    raise ValueError("Invalid native game")
                if g.get("truncated") is True and g.get("result_white") is not None:
                    raise ValueError("Manufactured draw for censored game")
                db.execute("INSERT INTO games VALUES (?,?,?,?,0)",
                           (g["game_id"],split(g["game_id"],seed),
                            json.dumps(g["moves"]),g["rows"]))
                gamecount+=1
        if gamecount!=manifest["games"]:raise ValueError("Game count mismatch")
        db.commit()
        positions=0
        counts=[0,0,0]
        removed=[0,0,0]
        # Three passes, train first. Keeps exact-input duplicates out of test
        # even when games are shuffled or repeated in source order.
        for part in range(3):
            with (data/"positions.jsonl").open() as handle:
                for line in handle:
                    r=json.loads(line)
                    if part==0:positions+=1
                    g=db.execute("SELECT part FROM games WHERE game_id=?",
                                 (r.get("game_id"),)).fetchone()
                    if g is None:raise ValueError("Position's game is missing")
                    if g[0]!=part:continue
                    if (r.get("schema")!=3 or type(r.get("side_to_move")) is not int
                        or r["side_to_move"] not in (0,1) or
                        type(r.get("ply")) is not int or not r.get("fen")):
                        raise ValueError("Invalid native position record")
                    label=r.get("teacher_stm_cp")
                    if (type(label) not in (float,int) or not -10000<=label<=10000):
                        raise ValueError("Invalid classical teacher label")
                    w=encode(r["features_white"],dim)
                    b=encode(r["features_black"],dim)
                    ident=hashlib.sha256(bytes((r["side_to_move"],))+w+b"\xff"+b).hexdigest()
                    bucket=int(ident[:2],16)%BUCKETS
                    cursor=db.execute("INSERT OR IGNORE INTO examples(input_key,part,bucket,white,black,stm,label,game_id,ply,fen) "
                                      "VALUES(?,?,?,?,?,?,?,?,?,?)",
                                      (ident,part,bucket,w,b,r["side_to_move"],
                                       float(np.clip(label,-2000,2000)),r["game_id"],r["ply"],r["fen"]))
                    if cursor.rowcount:counts[part]+=1
                    else:removed[part]+=1
                    db.execute("UPDATE games SET seen=seen+1 WHERE game_id=?",(r["game_id"],))
            db.commit()
        if positions!=manifest["positions"] or sum(counts)+sum(removed)!=positions:
            raise ValueError("Native position row counts differ from source")
        if db.execute("SELECT game_id FROM games WHERE seen!=expected LIMIT 1").fetchone():
            raise ValueError("Per-game position rows were missing")
        if min(counts)<8:raise ValueError("Too few genuine heldout or training examples")
        db.execute("CREATE INDEX sample_batch ON examples(part,bucket,id)")
        summary={"format":"v3-stream-1","descriptor":descriptor,"signature":signature,
                 "seed":seed,"teacher":manifest["teacher_options"],
                 "teacher_engine_sha256":manifest["native_engine_sha256"],
                 "input_rows":positions,"distinct_rows":sum(counts),
                 "partitions":dict(zip(PARTS,counts)),
                 "removed":dict(zip(PARTS,removed)),
                 "game_count":gamecount,"manifest_sha256":source}
        for k,v in (("signature",signature),("summary",canonical(summary))):
            db.execute("INSERT INTO metadata VALUES (?,?)",(k,v))
        db.commit()
        db.close()
        tmp.replace(out)
        return summary
    except BaseException:
        db.close()
        if tmp.exists():tmp.unlink()
        raise

def batches(db,part,batch_size,epoch_seed=0):
    order=np.random.default_rng(epoch_seed).permutation(BUCKETS)
    for bucket in order:
        last=0
        while True:
            found=db.execute("SELECT id,white,black,stm,label,game_id,ply,fen FROM examples "
                             "WHERE part=? AND bucket=? AND id>? ORDER BY id LIMIT ?",
                             (part,int(bucket),last,batch_size)).fetchall()
            if not found:break
            last=found[-1][0]
            yield found

def sparse_batch(rows,dim):
    def matrix(position):
        ri,ci,values=[],[],[]
        for i,item in enumerate(rows):
            for idx,count in unpack(item[position]):
                ri.append(i);ci.append(idx);values.append(count)
        return csr_matrix((np.asarray(values,dtype=np.float32),
                          (np.asarray(ri,dtype=np.int32),np.asarray(ci,dtype=np.int32))),
                          shape=(len(rows),dim),dtype=np.float32)
    white=matrix(1)
    black=matrix(2)
    side=np.asarray([r[3] for r in rows],dtype=np.int64)
    labels=np.asarray([r[4] for r in rows],dtype=np.float32)
    return white,black,side,labels
