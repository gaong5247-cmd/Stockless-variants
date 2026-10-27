"""Row-sparse AdaGrad feature transformer for million-row SVNNUE3 corpora.

The dense 128-wide hidden tail retains Adam; the first sparse feature
transformer updates only touched feature rows, without a full
(input_dimension x 128) gradient on every mini-batch. Sparse AdaGrad does
not apply momentum/weight updates to absent rows. This is deliberately
a different optimizer from the historical all-dense Adam checkpoints.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import numpy as np


def touched_gradient(white, black, grad_white, grad_black, dimension):
    """Return sorted touched row IDs and exactly accumulated FP32 gradients."""
    if not hasattr(white, "tocsr") or not hasattr(black, "tocsr"):
        raise ValueError("Sparse FT optimization requires native CSR matrices")
    if (white.shape != black.shape or white.shape[1] != dimension
        or grad_white.shape != grad_black.shape
        or grad_white.shape != (white.shape[0], 128)):
        raise ValueError("Incompatible native dual-perspective gradient shape")
    indices=np.union1d(white.indices,black.indices)
    if not len(indices):
        raise ValueError("Empty royal feature-transformer minibatch")
    # Slice columns before sparse-transpose matmul. The gradient tensor has
    # touched_rows x 128 floats rather than all input rows x 128 floats.
    selected_white=white[:,indices]
    selected_black=black[:,indices]
    gradient=np.asarray(selected_white.T@grad_white+
                        selected_black.T@grad_black,dtype=np.float32)
    if gradient.shape != (len(indices),128) or not np.isfinite(gradient).all():
        raise ValueError("Non-finite or wrong-size sparse transformer gradient")
    return indices,gradient


def update_touched_adagrad(weights,accumulated,indices,gradient,
                           learning_rate,regularization=1e-7,clip=8.):
    """In-place rowwise AdaGrad; the rest of the matrix is bitwise unchanged."""
    if (gradient.shape != (len(indices),weights.shape[1]) or
        weights.shape != accumulated.shape):
        raise ValueError("Sparse optimizer dimensions do not agree")
    if not len(indices):
        return
    old=weights[indices]
    grad=np.asarray(gradient,dtype=np.float32)+regularization*old
    if not np.isfinite(grad).all():
        raise ValueError("Non-finite input to sparse AdaGrad")
    moments=accumulated[indices]+grad*grad
    # Unlike dense Adam, inactive rows do not move between activations.
    updated=old-learning_rate*grad/(np.sqrt(moments)+1e-8)
    if not np.isfinite(updated).all():
        raise ValueError("Non-finite transformer update")
    np.clip(updated,-clip,clip,out=updated)
    accumulated[indices]=moments
    weights[indices]=updated
