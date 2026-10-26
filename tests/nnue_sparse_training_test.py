#!/usr/bin/env python3
"""Compare CSR/native-sparse vs dense NNUE on identical batches.

CPU benchmark numbers are diagnostic and not interpreted as engine Elo.
SPDX-License-Identifier: AGPL-3.0-only
"""
import json
import os
from pathlib import Path
import struct
import sys
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/nnue"))
from model import float_predict, integer_predict, quantize
from sparse_batches import csr_batch, forward, integer_forward, first_layer_gradient
from stream_dataset import dense_batch


def fixture(variant, width, height, types, seed=64):
    d = {"supported": True, "variant": variant, "format_version": 2,
         "feature_version": 2, "width": width, "height": height,
         "type_ids": types, "hidden_dim": 64, "qa": 256, "qb": 64,
         "rules_sha256": "1"*64}
    area = width * height
    base = 2 * len(types) * (area + 1)
    d["input_dim"] = base + 2 * len(types) * area + 4 + width + 6 * len(types)
    rng = np.random.default_rng(seed)
    weights = {
        "w1": rng.normal(0, .02, (d["input_dim"],64)).astype(np.float32),
        "b1": np.full(64, .25, dtype=np.float32),
        "w2": rng.normal(0, 3, (2,64)).astype(np.float32),
        "b2": np.zeros(2, dtype=np.float32),
    }
    records = []
    for j in range(256):
        features = np.sort(rng.choice(d["input_dim"], size=48, replace=False))
        blob = b"".join(struct.pack("<HH", int(i), 1) for i in features)
        records.append((j, blob, j % 2, float(rng.normal(0,300)), "game",j,"dummy-fen"))
    return d, weights, records


def check(variant, width, height, types):
    d,w,records = fixture(variant,width,height,types)
    dense, side, y = dense_batch(records, d["input_dim"])
    sparse, side2, y2 = csr_batch(records, d["input_dim"])
    assert np.array_equal(side,side2) and np.array_equal(y,y2)
    np.testing.assert_array_equal(dense,sparse.toarray())
    np.testing.assert_allclose(float_predict(w,dense,side),forward(w,sparse,side),
                               rtol=1e-6,atol=0.001)
    q = quantize(w,d)
    np.testing.assert_array_equal(integer_predict(q,dense.astype(np.int64),side),
                                  integer_forward(q,sparse,side))
    rng=np.random.default_rng(13)
    hidden=rng.normal(0,1,(len(records),64)).astype(np.float32)
    np.testing.assert_allclose(dense.T @ hidden,first_layer_gradient(sparse,hidden),
                               rtol=1e-5,atol=0.0002)
    # Warm up memory allocation and BLAS kernels; time identical operations.
    for _ in range(2):
        dense@w["w1"]; dense.T@hidden
        sparse@w["w1"]; sparse.T@hidden
    times={}
    for label, matrix in (("dense",dense),("sparse_csr",sparse)):
        start=perf_counter()
        for _ in range(10):
            forward1 = matrix@w["w1"]
            grad1 = matrix.T@hidden
        times[label]=(perf_counter()-start)/10
        assert np.isfinite(forward1).all() and np.isfinite(grad1).all()
    report={"variant":variant,"rows":len(records),"input_dim":d["input_dim"],
            "nonzero_features_per_row":48,
            "dense_seconds_per_step":times["dense"],
            "sparse_seconds_per_step":times["sparse_csr"],
            "speedup_sparse_vs_dense":times["dense"]/times["sparse_csr"],
            "exact_integer_parity":True,"float_gradient_parity":True,
            "openblas_threads":os.getenv("OPENBLAS_NUM_THREADS")}
    print(json.dumps(report,sort_keys=True),flush=True)
    return report


if __name__=="__main__":
    cases=[
        ("chess",8,8,[1,2,3,4,5,63]),
        ("shogi",9,9,[3,4,9,11,19,20,21,22,23,63]),
        ("janggi",9,10,[4,28,29,30,32,34,63]),
        ("xiangqi",9,10,[4,6,27,29,30,31,63]),
        ("makruk",8,8,[1,2,4,6,9,63])
    ]
    results=[check(*case) for case in cases]
    aggregate=sum(r["dense_seconds_per_step"] for r in results)/sum(r["sparse_seconds_per_step"] for r in results)
    print(json.dumps({"cases":len(results),"cross_variant_speedup":aggregate,
                      "note":"This is a CPU transformer-training benchmark, not an engine NPS or Elo measurement."}))
