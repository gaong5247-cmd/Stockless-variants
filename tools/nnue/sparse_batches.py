"""Sparse CPU minibatches for native SVNNUE training.

Convert only native (index, count) pairs in a bounded batch to CSR.
Forward and backward avoid the full input_dim matrix for each batch.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from model import QA, QB
from stream_dataset import decode_features


def csr_batch(records, dimension):
    rows, cols, vals = [], [], []
    sides = np.empty(len(records), dtype=np.int64)
    labels = np.empty(len(records), dtype=np.float32)
    for i, (_, encoded, side, target, *_rest) in enumerate(records):
        sides[i] = side
        labels[i] = target
        for idx, value in decode_features(encoded):
            if not 0 <= idx < dimension or not 1 <= value <= 512:
                raise ValueError("Invalid native sparse feature in indexed training data")
            rows.append(i)
            cols.append(idx)
            vals.append(value)
    matrix = csr_matrix(
        (np.asarray(vals, dtype=np.float32),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=(len(records), dimension), dtype=np.float32,
    )
    return matrix, sides, labels


def forward(weights, matrix, side):
    activation = np.clip(matrix @ weights["w1"] + weights["b1"], 0., 1.)
    return np.sum(activation * weights["w2"][side], axis=1) + weights["b2"][side]


def integer_forward(weights, matrix, side):
    accumulator = matrix.astype(np.int64) @ weights["w1"].astype(np.int64) + weights["b1"].astype(np.int64)
    if np.max(np.abs(accumulator), initial=0) >= 1 << 31:
        raise ValueError("Native NNUE accumulator exceeds int32 bounds")
    active = np.clip(accumulator, 0, QA)
    numerator = np.sum(active * weights["w2"][side].astype(np.int64), axis=1) + weights["b2"][side].astype(np.int64)
    denom = QA * QB
    result = np.sign(numerator) * ((np.abs(numerator) + denom // 2) // denom)
    return np.clip(result, -10000, 10000).astype(np.int64)


def first_layer_gradient(matrix, hidden):
    return np.asarray(matrix.T @ hidden, dtype=np.float32)
