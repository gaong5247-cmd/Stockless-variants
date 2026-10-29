#!/usr/bin/env python3
"""Unit gate for sparse FT AdaGrad vs full dense source gradient and full restart.

Unlike Adam over the entire input FT, unused rows MUST remain bitwise
unchanged and receive no artificial momentum drift.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import tempfile
import sqlite3

import numpy as np
from scipy.sparse import csr_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"/"nnue"))
from sparse_ft import touched_gradient,update_touched_adagrad
from train_v3 import state_init,optimizer_step,store,restore,KEYS
from model_v3 import initialize

def run():
    rng=np.random.default_rng(821)
    dimension=96
    hidden=128
    rows=16
    white=csr_matrix((rng.random((rows,dimension))<.085).astype(np.float32))
    black=csr_matrix((rng.random((rows,dimension))<.080).astype(np.float32))
    gw=rng.normal(0,1,(rows,hidden)).astype(np.float32)
    gb=rng.normal(0,1,(rows,hidden)).astype(np.float32)
    idx,g=touched_gradient(white,black,gw,gb,dimension)
    dense=white.toarray().T@gw+black.toarray().T@gb
    np.testing.assert_allclose(g,dense[idx],rtol=1e-5,atol=2e-5)
    assert 0<len(idx)<dimension
    initial=rng.normal(0,.1,(dimension,hidden)).astype(np.float32)
    accumulated=np.zeros_like(initial)
    first=initial.copy()
    update_touched_adagrad(first,accumulated,idx,g,.01)
    untouched=np.setdiff1d(np.arange(dimension),idx)
    assert np.array_equal(first[untouched],initial[untouched])
    assert np.count_nonzero(accumulated[untouched])==0
    assert np.all(accumulated[idx]>=0)

    d={"schema":3,"supported":True,"format_version":3,"feature_version":3,
       "variant":"chess","width":8,"height":8,"type_ids":[1,2,3,4,5,63],
       "input_dim":16*2*6*65,"hidden_dim":128,"l1_dim":32,"l2_dim":16,
       "qa":256,"qb":128,"king_buckets":16,"black_rotation":False,
       "rules_sha256":"0"*64}
    with tempfile.TemporaryDirectory() as folder:
        a=Path(folder)/"continuous"
        b=Path(folder)/"resume"
        a.mkdir()
        b.mkdir()
        with sqlite3.connect(":memory:") as db:
            db.execute("CREATE TABLE examples(part INTEGER,label REAL)")
            db.execute("INSERT INTO examples VALUES(0,0.0)")
            state_a=state_init(d,db,51,"sparse_ft_adagrad")
            state_b=state_init(d,db,51,"sparse_ft_adagrad")
            side=np.asarray([i%2 for i in range(rows)],dtype=np.int64)
            target=rng.normal(0,40,rows).astype(np.float32)
            # Native CSR matrix must use feature indices within v3 input dim.
            p=white.indices.copy()
            q=black.indices.copy()
            xx=csr_matrix((white.data,white.indices,white.indptr),shape=(rows,d["input_dim"]))
            yy=csr_matrix((black.data,black.indices,black.indptr),shape=(rows,d["input_dim"]))
            for _ in range(2):
                optimizer_step(state_a,xx,yy,side,target,1)
            optimizer_step(state_b,xx,yy,side,target,1)
            state_b["epoch"]=1
            store(b,state_b,"test-dataset-sig",51,16)
            resumed=restore(b,"test-dataset-sig",51,16,d,"sparse_ft_adagrad")
            optimizer_step(resumed,xx,yy,side,target,1)
            for key in KEYS:
                for name in ("weights","moment","variance"):
                    if not np.array_equal(state_a[name][key],resumed[name][key]):
                        raise AssertionError("Sparse exact optimizer resume mismatch: "+name+"/"+key)
            try:
                restore(b,"other-data",51,16,d,"sparse_ft_adagrad")
                raise AssertionError("Accepted dataset drift")
            except ValueError:
                pass
            try:
                restore(b,"test-dataset-sig",51,16,d,"dense_adam")
                raise AssertionError("Accepted different optimizer state")
            except ValueError:
                pass
    print(json.dumps({"sparse_rows":len(idx),"total_input_rows":dimension,
          "dense_gradient_parity":True,"unused_weights_bit_exact":True,
          "complete_optimizer_restart_identical":True,
          "changed_dataset_or_optimizer_rejected":True}))

if __name__=="__main__":
    run()
