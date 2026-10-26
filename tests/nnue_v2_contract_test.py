#!/usr/bin/env python3
"""Contract checks for independent SVNNUE-v1/v2 model layouts and magic."""
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"/"nnue"))
from model import HIDDEN, MAGIC, MAGIC_V2, validate_descriptor, quantize, write_network, read_network, integer_predict


def descriptor(version,variant,width,height,types):
    area=width*height
    base=2*len(types)*(area+1)
    dimension=base if version==1 else base+2*len(types)*area+4+width+6*len(types)
    return {"supported":True,"format_version":version,"feature_version":version,
            "variant":variant,"width":width,"height":height,"type_ids":types,
            "hidden_dim":HIDDEN,"input_dim":dimension,"qa":256,"qb":64,
            "rules_sha256":"a"*64}


def main():
    dims=[
        ("chess",8,8,[1,2,3,4,5,63]),
        ("shogi",9,9,[3,4,9,11,19,20,21,22,23,63]),
        ("janggi",9,10,[4,28,29,30,32,34,63]),
        ("xiangqi",9,10,[4,6,27,29,30,31,63]),
        ("makruk",8,8,[1,2,4,6,9,63]),
    ]
    rng=np.random.default_rng(1234)
    with TemporaryDirectory() as td:
        for variant,w,h,types in dims:
            for version in (1,2):
                d=descriptor(version,variant,w,h,types)
                validate_descriptor(d)
                weights={"w1":rng.normal(0,.025,(d["input_dim"],HIDDEN)).astype(np.float32),
                         "b1":np.full(HIDDEN,.25,dtype=np.float32),
                         "w2":rng.normal(0,4,(2,HIDDEN)).astype(np.float32),
                         "b2":np.zeros(2,dtype=np.float32)}
                q=quantize(weights,d)
                path=Path(td)/(variant+"-v"+str(version)+".svnnue")
                write_network(path,d,q)
                assert path.read_bytes()[:8]==(MAGIC_V2 if version==2 else MAGIC)
                loaded=read_network(path,d)
                for key in q:
                    assert np.array_equal(q[key],loaded[key])
                x=np.zeros((4,d["input_dim"]),dtype=np.int64)
                x[0,0]=1
                x[1,d["input_dim"]-1]=2
                x[2,5]=1
                sides=np.array([0,1,0,1],dtype=np.int64)
                assert np.array_equal(integer_predict(q,x,sides),integer_predict(loaded,x,sides))
                wrong=dict(d,format_version=3,feature_version=3)
                try:
                    validate_descriptor(wrong)
                    raise AssertionError("Accepted unknown NNUE version")
                except ValueError:
                    pass
                other=dict(d,rules_sha256="b"*64)
                try:
                    read_network(path,other)
                    raise AssertionError("Accepted wrong rule fingerprint")
                except ValueError:
                    pass
                print("PASS",variant,"SVNNUE-v"+str(version),d["input_dim"],"features")


if __name__=="__main__":
    main()
