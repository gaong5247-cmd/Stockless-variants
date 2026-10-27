"""Independent portable SVNNUE-v3 Lite network: two royal perspectives.

The custom .svnnue is NOT an upstream Stockfish or Fairy-Stockfish net.
FP32 model and fully specified int32 fixed-point inference share one schema.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import struct
import numpy as np

MAGIC=b"SVNNUE3\x00"
HEADER_BYTES=160
HIDDEN=128
L1=32
L2=16
BUCKETS=16
QA=256
QB=128
MAX_FILE_BYTES=32*1024*1024
ALLOWED=("chess","shogi","janggi","xiangqi","makruk")

SHAPES=(("ft",None,"<i2",QA),("bft",(HIDDEN,),"<i4",QA),
        ("w1",(2*HIDDEN,L1),"<i2",QB),("b1",(L1,),"<i4",QA*QB),
        ("w2",(L1,L2),"<i2",QB),("b2",(L2,),"<i4",QA*QB),
        ("w3",(L2,),"<i2",QB),("b3",(1,),"<i4",QA*QB))

def digest(path:Path)->str:
    hasher=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda:stream.read(1024*1024),b""):
            hasher.update(data)
    return hasher.hexdigest()

def validate_descriptor(d):
    if (not isinstance(d,dict) or d.get("schema")!=3
        or d.get("format_version")!=3 or d.get("feature_version")!=3
        or d.get("variant") not in ALLOWED or d.get("supported") is not True):
        raise ValueError("Unsupported SVNNUE-v3 variant/format")
    w,h,types=d.get("width"),d.get("height"),d.get("type_ids")
    if (type(w) is not int or type(h) is not int or not (1<=w<=12 and 1<=h<=10)
        or not isinstance(types,list) or not types or
        types!=sorted(set(types)) or any(type(t) is not int or not 1<=t<=63 for t in types)):
        raise ValueError("Invalid board geometry or native piece types")
    dim=BUCKETS*2*len(types)*(w*h+1)
    required={"input_dim":dim,"hidden_dim":HIDDEN,"l1_dim":L1,"l2_dim":L2,
              "qa":QA,"qb":QB,"king_buckets":BUCKETS,
              "black_rotation":d["variant"]=="shogi"}
    for key,value in required.items():
        if d.get(key)!=value:raise ValueError(f"Wrong {key} in v3 descriptor")
    rule=d.get("rules_sha256")
    if not isinstance(rule,str) or len(rule)!=64:
        raise ValueError("Missing rule fingerprint")
    try:bytes.fromhex(rule)
    except ValueError as e: raise ValueError("Invalid rule fingerprint") from e
    return dim

def shapes(d):
    dim=validate_descriptor(d)
    return {key:(dim,HIDDEN) if key=="ft" else shape for key,shape,_,_ in SHAPES}

def initialize(d,seed,offset_cp=0.):
    dim=validate_descriptor(d)
    rng=np.random.default_rng(seed)
    return {
        "ft":rng.normal(0,.012,(dim,HIDDEN)).astype(np.float32),
        "bft":np.full(HIDDEN,.25,dtype=np.float32),
        "w1":rng.normal(0,.08,(2*HIDDEN,L1)).astype(np.float32),
        "b1":np.full(L1,.25,dtype=np.float32),
        "w2":rng.normal(0,.08,(L1,L2)).astype(np.float32),
        "b2":np.full(L2,.25,dtype=np.float32),
        "w3":rng.normal(0,8,(L2,)).astype(np.float32),
        "b3":np.asarray([offset_cp],dtype=np.float32)
    }

def activations(w,white,black,side):
    aw=np.clip(white@w["ft"]+w["bft"],0,1)
    ab=np.clip(black@w["ft"]+w["bft"],0,1)
    left=np.where(side[:,None]==0,aw,ab)
    right=np.where(side[:,None]==0,ab,aw)
    stack=np.concatenate((left,right),axis=1)
    z1=stack@w["w1"]+w["b1"]
    h1=np.clip(z1,0,1)
    z2=h1@w["w2"]+w["b2"]
    h2=np.clip(z2,0,1)
    return aw,ab,stack,z1,h1,z2,h2

def float_predict(w,white,black,side):
    *_,h2=activations(w,white,black,side)
    return np.asarray(h2@w["w3"]+w["b3"][0],dtype=np.float32)

def quantize(w,d):
    expected=shapes(d)
    result={}
    for key,_,dtype,scale in SHAPES:
        a=np.asarray(w[key],dtype=np.float64)
        if a.shape!=expected[key] or not np.isfinite(a).all():
            raise ValueError(f"Invalid fp32 tensor {key}")
        q=np.copysign(np.floor(np.abs(a*scale)+.5),a)
        if dtype=="<i2":
            if np.any(q< -32768) or np.any(q>32767):
                raise ValueError(f"int16 weight overflow: {key}")
        elif key=="bft":
            if np.any(np.abs(q)>1<<20):
                raise ValueError("Transformer bias would overflow native bounds")
        elif np.any(q<-(1<<31)) or np.any(q>(1<<31)-1):
            raise ValueError(f"int32 bias overflow: {key}")
        result[key]=q.astype(dtype)
    return result

def signed_round(value,divisor):
    x=np.asarray(value,dtype=np.int64)
    return np.sign(x)*((np.abs(x)+divisor//2)//divisor)

def integer_predict(q,white,black,side):
    # Preserve CSR sparsity: converting an entire sparse batch with
    # np.asarray produces an object scalar and fails integer reference math.
    w=white.astype(np.int64) if hasattr(white,"tocsr") else np.asarray(white,dtype=np.int64)
    b=black.astype(np.int64) if hasattr(black,"tocsr") else np.asarray(black,dtype=np.int64)
    ft=q["ft"].astype(np.int64)
    bias=q["bft"].astype(np.int64)
    aw=w@ft+bias
    ab=b@ft+bias
    if max(np.max(np.abs(aw),initial=0),np.max(np.abs(ab),initial=0)) >= (1<<31):
        raise ValueError("Transformer overflow")
    aa=np.clip(aw,0,QA)
    bb=np.clip(ab,0,QA)
    stack=np.concatenate([np.where(side[:,None]==0,aa,bb),
                          np.where(side[:,None]==0,bb,aa)],axis=1)
    h1=np.clip(signed_round(stack@q["w1"].astype(np.int64)+q["b1"].astype(np.int64),QB),0,QA)
    h2=np.clip(signed_round(h1@q["w2"].astype(np.int64)+q["b2"].astype(np.int64),QB),0,QA)
    scores=signed_round(h2@q["w3"].astype(np.int64)+q["b3"].astype(np.int64)[0],QA*QB)
    return np.clip(scores,-10000,10000).astype(np.int64)

def write_network(path,d,q):
    dim=validate_descriptor(d)
    payload=bytearray()
    for key,_,dtype,_ in SHAPES:
        val=np.asarray(q[key],dtype=dtype)
        if val.shape!=shapes(d)[key]:raise ValueError(f"Tensor shape mismatch {key}")
        payload.extend(val.tobytes(order="C"))
    if len(payload)+HEADER_BYTES>MAX_FILE_BYTES:
        raise ValueError("Network exceeds native bounded file size")
    head=bytearray(HEADER_BYTES)
    head[:8]=MAGIC
    struct.pack_into("<10I",head,8,3,3,HIDDEN,dim,d["width"],d["height"],
                     len(d["type_ids"]),QA,QB,len(payload))
    name=d["variant"].encode("ascii")
    head[48:48+len(name)]=name
    head[80:112]=bytes.fromhex(d["rules_sha256"])
    head[112:144]=hashlib.sha256(payload).digest()
    dest=Path(path)
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix(dest.suffix+".tmp")
    temp.write_bytes(bytes(head)+bytes(payload))
    temp.replace(dest)

def read_network(path,d):
    dim=validate_descriptor(d)
    data=Path(path).read_bytes()
    if not HEADER_BYTES<=len(data)<=MAX_FILE_BYTES or data[:8]!=MAGIC:
        raise ValueError("SVNNUE3 magic or bounds mismatch")
    expected=(3,3,HIDDEN,dim,d["width"],d["height"],len(d["type_ids"]),
              QA,QB,len(data)-HEADER_BYTES)
    if struct.unpack_from("<10I",data,8)!=expected:
        raise ValueError("Network v3 geometry, dimensions or version mismatch")
    name=d["variant"].encode("ascii")
    if (data[48:80]!=name+bytes(32-len(name))
        or data[80:112]!=bytes.fromhex(d["rules_sha256"])
        or any(data[144:160])):
        raise ValueError("Network rule/variant identity mismatch")
    payload=data[HEADER_BYTES:]
    if hashlib.sha256(payload).digest()!=data[112:144]:
        raise ValueError("Network payload SHA256 mismatch")
    off=0
    result={}
    for key,shape,dtype,_ in SHAPES:
        actual=(dim,HIDDEN) if key=="ft" else shape
        count=int(np.prod(actual))
        a=np.frombuffer(payload,dtype=dtype,count=count,offset=off)
        result[key]=a.reshape(actual).copy()
        off+=count*np.dtype(dtype).itemsize
    if off!=len(payload) or np.max(np.abs(result["bft"].astype(np.int64)))>1<<20:
        raise ValueError("Corrupt v3 model payload")
    return result
