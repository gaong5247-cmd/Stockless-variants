#!/usr/bin/env python3
"""SVNNUE3 explicit fixed-point precision contract.

Both QA=256 legacy and QA=2048 hi-resolution networks share the same
royal-feature indices and different signed, SHA-authenticated headers.
No data or model strength is invented by this numeric-format test.
"""
from pathlib import Path
import sys
import tempfile
import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"/"nnue"))
from model_v3 import initialize,quantize,read_network,write_network,integer_predict,float_predict

def main():
    d={"schema":3,"supported":True,"format_version":3,"feature_version":3,
       "variant":"chess","width":8,"height":8,"type_ids":[1,2,3,4,5,63],
       "input_dim":16*2*6*(8*8+1),"hidden_dim":128,
       "l1_dim":32,"l2_dim":16,"qa":256,"qb":128,
       "king_buckets":16,"black_rotation":False,
       "rules_sha256":"1"*64}
    w=initialize(d,20261027)
    rng=np.random.default_rng(20261027)
    samples=72
    rows=[];cols=[]
    for i in range(samples):
        active=rng.choice(d["input_dim"],size=32,replace=False)
        for col in active:
            rows.append(i);cols.append(int(col))
    X=csr_matrix((np.ones(len(rows),dtype=np.float32),
                  (rows,cols)),shape=(samples,d["input_dim"]))
    Y=X[:,::-1].tocsr()
    side=np.asarray([i%2 for i in range(samples)],dtype=np.int64)
    reference=float_predict(w,X,Y,side)
    comparison={}
    with tempfile.TemporaryDirectory() as tmp:
        for qa in (256,512,1024,2048,4096):
            desc=dict(d,qa=qa)
            q=quantize(w,desc)
            assert q["__qa__"]==qa and q["__qb__"]==128
            file=Path(tmp)/f"test-{qa}.svnnue"
            write_network(file,desc,q)
            net=read_network(file,desc)
            got=integer_predict(net,X,Y,side)
            assert got.dtype==np.int64 and np.all(np.isfinite(got))
            assert np.array_equal(integer_predict(q,X,Y,side),got)
            bad=dict(desc,qa=(qa*2 if qa<4096 else 256))
            try:
                read_network(file,bad)
                raise AssertionError("Cross-precision model must be rejected")
            except ValueError:
                pass
            comparison[qa]=float(np.mean(np.abs(got-reference)))
    assert comparison[2048]<comparison[256],comparison
    print("PASS SVNNUE3 native quantization profiles, original model compatibility",
          "QA256 raw MAE",comparison[256],"QA2048 raw MAE",comparison[2048])

if __name__=="__main__":
    main()
