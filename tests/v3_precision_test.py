#!/usr/bin/env python3
"""Versioned SVNNUE3Q fixed-point precision roundtrip and safety regression.

V3Q changes only quantization, NOT royal feature indices or model shape.
Legacy networks remain byte compatible and are independently loadable.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"/"nnue"))
from model_v3 import MAGIC,MAGIC_Q, initialize,quantize,read_network,write_network,integer_predict,float_predict,PRECISION_MODES,shapes

def desc(name,width,height,types):
    area=width*height
    return {"schema":3,"supported":True,"format_version":3,"feature_version":3,
            "variant":name,"width":width,"height":height,"type_ids":types,
            "input_dim":16*2*len(types)*(area+1),"hidden_dim":128,
            "l1_dim":32,"l2_dim":16,"qa":256,"qb":128,"king_buckets":16,
            "black_rotation":name=="shogi","rules_sha256":"ab"*32}

def main():
    cases=[
        ("chess",8,8,[1,2,3,4,5,63]),
        ("shogi",9,9,[3,4,9,11,19,20,21,22,23,63]),
        ("janggi",9,10,[4,28,29,30,32,34,63]),
        ("xiangqi",9,10,[4,6,27,29,30,31,63]),
        ("makruk",8,8,[1,2,4,6,9,63])]
    assert PRECISION_MODES=={"legacy":(1,1),"high":(4,2)}
    rng=np.random.default_rng(20261027)
    with TemporaryDirectory() as td:
        for name,w,h,types in cases:
            d=desc(name,w,h,types)
            model=initialize(d,seed=31415)
            dim=shapes(d)["ft"][0]
            nw=24
            sid=rng.choice(dim,size=24*nw,replace=True)
            srow=np.repeat(np.arange(nw),24)
            wcsr=csr_matrix((np.ones(len(sid),dtype=np.float32),(srow,sid)),
                            shape=(nw,dim),dtype=np.float32)
            bid=rng.choice(dim,size=25*nw,replace=True)
            brow=np.repeat(np.arange(nw),25)
            bcsr=csr_matrix((np.ones(len(bid),dtype=np.float32),(brow,bid)),
                            shape=(nw,dim),dtype=np.float32)
            sides=np.arange(nw,dtype=np.int64)%2
            errors={}
            for precision in ("legacy","high"):
                q=quantize(model,d,precision_mode=precision)
                file=Path(td)/(name+"-"+precision+".svnnue")
                write_network(file,d,q)
                magic=MAGIC_Q if precision=="high" else MAGIC
                assert file.read_bytes()[:8]==magic
                reread=read_network(file,d)
                for key in ("ft","bft","w1","b1","w2","b2","w3","b3"):
                    assert np.array_equal(q[key],reread[key]),key
                a=integer_predict(q,wcsr,bcsr,sides)
                b=integer_predict(reread,wcsr,bcsr,sides)
                assert np.array_equal(a,b)
                expected=float_predict(model,wcsr,bcsr,sides)
                errors[precision]=float(np.mean(np.abs(a-expected)))
                # Mixed-format endian/magic corruption is always rejected.
                bad=file.with_name(file.stem+"-bad.svnnue")
                packed=bytearray(file.read_bytes())
                packed[-1]^=1
                bad.write_bytes(packed)
                try:
                    read_network(bad,d)
                    raise AssertionError("Accepted wrong SHA256 payload")
                except ValueError:
                    pass
            print(f"{name}: legacy MAE {errors['legacy']:.3f} cp, "
                  f"high-precision MAE {errors['high']:.3f} cp; "
                  "both cryptographically bound networks loaded")
    print("PASS versioned v3/3Q fixed point roundtrip + corruption gates")

if __name__=="__main__":
    main()
