"""Independent SVNNUE-v1 training/export format; not a Stockfish .nnue file.

Copyright (C) 2026 Stockless Variants contributors
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct
from typing import Mapping

import numpy as np

MAGIC = b"SVNNUE1\0"
MAGIC_V2 = b"SVNNUE2\0"
HEADER_BYTES = 160
HIDDEN = 64
QA = 256
QB = 64
FORMAT_VERSION = 1
FEATURE_VERSION = 1
MAX_FILE_BYTES = 4 * 1024 * 1024
VARIANTS = ("chess", "shogi", "janggi", "xiangqi", "makruk")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_descriptor(d: Mapping) -> None:
    if d.get("supported") is not True or d.get("variant") not in VARIANTS:
        raise ValueError("SVNNUE-v1 supports only the five audited native variants")
    version = d.get("format_version")
    if type(version) is not int or version not in (1, 2):
        raise ValueError("Unsupported SVNNUE network format")
    expected = {"format_version": version, "feature_version": version,
                "hidden_dim": HIDDEN, "qa": QA, "qb": QB}
    for name, value in expected.items():
        if d.get(name) != value:
            raise ValueError(f"Incompatible {name}: {d.get(name)!r}")
    width, height, types = d["width"], d["height"], d["type_ids"]
    if type(width) is not int or type(height) is not int or not (1 <= width <= 12 and 1 <= height <= 10):
        raise ValueError("Invalid native board geometry")
    if not types or types != sorted(set(types)) or any(type(t) is not int or not 1 <= t <= 63 for t in types):
        raise ValueError("Piece types must be distinct, sorted native type IDs")
    area = width * height
    base = 2 * len(types) * (area + 1)
    expected_dim = base if version == 1 else base + 2 * len(types) * area + 4 + width + 6 * len(types)
    if d["input_dim"] != expected_dim:
        raise ValueError("Feature dimension does not match native SVNNUE-v1/v2 layout")
    rule_hash = d.get("rules_sha256", "")
    if len(rule_hash) != 64 or len(bytes.fromhex(rule_hash)) != 32:
        raise ValueError("Missing 256-bit rules identity")


def round_away(values: np.ndarray) -> np.ndarray:
    return np.copysign(np.floor(np.abs(values) + 0.5), values)


def float_predict(weights: Mapping[str, np.ndarray], x: np.ndarray, side: np.ndarray) -> np.ndarray:
    activation = np.clip(x @ weights["w1"] + weights["b1"], 0.0, 1.0)
    return np.sum(activation * weights["w2"][side], axis=1) + weights["b2"][side]


def quantize(weights: Mapping[str, np.ndarray], d: Mapping) -> dict[str, np.ndarray]:
    validate_descriptor(d)
    shapes = {"w1": (d["input_dim"], HIDDEN), "b1": (HIDDEN,),
              "w2": (2, HIDDEN), "b2": (2,)}
    scales = {"w1": QA, "b1": QA, "w2": QB, "b2": QA * QB}
    result = {}
    for key, shape in shapes.items():
        array = np.asarray(weights[key], dtype=np.float64)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid floating-point {key}")
        rounded = round_away(array * scales[key])
        # Match the runtime's accumulator proof and exact integer format.
        maximum = (1 << 20) if key == "b1" else ((1 << 31) - 1 if key == "b2" else 32767)
        minimum = -maximum if key == "b1" else (-(1 << 31) if key == "b2" else -32768)
        if np.any(rounded < minimum) or np.any(rounded > maximum):
            raise ValueError(f"{key} exceeds the SVNNUE-v1 quantization bounds")
        result[key] = rounded.astype("<i4" if key in ("b1", "b2") else "<i2")
    return result


def integer_predict(q: Mapping[str, np.ndarray], x: np.ndarray, side: np.ndarray) -> np.ndarray:
    # Int64 is deliberate: independent reference arithmetic must not inherit a
    # platform's narrowing overflow before checking the runtime's int32 bounds.
    x64 = np.asarray(x, dtype=np.int64)
    acc = x64 @ q["w1"].astype(np.int64) + q["b1"].astype(np.int64)
    if np.max(np.abs(acc), initial=0) >= (1 << 31):
        raise ValueError("Accumulator exceeds int32 bounds")
    act = np.clip(acc, 0, QA)
    numerator = np.sum(act * q["w2"][side].astype(np.int64), axis=1) + q["b2"][side].astype(np.int64)
    denominator = QA * QB
    rounded = np.sign(numerator) * ((np.abs(numerator) + denominator // 2) // denominator)
    return np.clip(rounded, -10000, 10000).astype(np.int64)


def write_network(path: Path, d: Mapping, q: Mapping[str, np.ndarray]) -> None:
    validate_descriptor(d)
    payload = b"".join(np.asarray(q[key], dtype="<i4" if key in ("b1", "b2") else "<i2").tobytes(order="C")
                       for key in ("w1", "b1", "w2", "b2"))
    header = bytearray(HEADER_BYTES)
    version = d["format_version"]
    header[:8] = MAGIC_V2 if version == 2 else MAGIC
    struct.pack_into("<10I", header, 8, version, version, HIDDEN,
                     d["input_dim"], d["width"], d["height"], len(d["type_ids"]), QA, QB, len(payload))
    name = d["variant"].encode("ascii")
    if len(name) > 31:
        raise ValueError("Variant ID exceeds header field")
    header[48:48 + len(name)] = name
    header[80:112] = bytes.fromhex(d["rules_sha256"])
    header[112:144] = hashlib.sha256(payload).digest()
    if len(header) + len(payload) > MAX_FILE_BYTES:
        raise ValueError("Network exceeds runtime file-size limit")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(bytes(header) + payload)
    temporary.replace(path)


def read_network(path: Path, d: Mapping) -> dict[str, np.ndarray]:
    validate_descriptor(d)
    data = path.read_bytes()
    expected_magic = MAGIC_V2 if d["format_version"] == 2 else MAGIC
    if not HEADER_BYTES <= len(data) <= MAX_FILE_BYTES or data[:8] != expected_magic:
        raise ValueError("Invalid SVNNUE file magic/length")
    values = struct.unpack_from("<10I", data, 8)
    expected = (d["format_version"], d["feature_version"], HIDDEN, d["input_dim"], d["width"], d["height"],
                len(d["type_ids"]), QA, QB, len(data) - HEADER_BYTES)
    if values != expected or data[144:160] != bytes(16):
        raise ValueError("Incompatible SVNNUE header")
    name = d["variant"].encode("ascii")
    if data[48:80] != name + bytes(32 - len(name)) or data[80:112] != bytes.fromhex(d["rules_sha256"]):
        raise ValueError("Network belongs to different variant/rules")
    payload = data[HEADER_BYTES:]
    if hashlib.sha256(payload).digest() != data[112:144]:
        raise ValueError("Corrupt network payload")
    offset = 0
    q = {}
    for key, shape, dtype in (("w1", (d["input_dim"], HIDDEN), "<i2"), ("b1", (HIDDEN,), "<i4"),
                              ("w2", (2, HIDDEN), "<i2"), ("b2", (2,), "<i4")):
        count = int(np.prod(shape))
        q[key] = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape).copy()
        offset += count * np.dtype(dtype).itemsize
    if offset != len(payload) or np.max(np.abs(q["b1"].astype(np.int64)), initial=0) > (1 << 20):
        raise ValueError("Invalid payload dimensions or accumulator bias")
    return q
