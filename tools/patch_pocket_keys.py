#!/usr/bin/env python3
"""Prevent out-of-bounds Zobrist reads for virtual partner-board reserves.

Apply after patch_rules.apply(). Existing nonnegative pocket hashes are retained
exactly. FSF's internal partner search can borrow pieces, producing negative
reserve counts even without an external partner connection.

Copyright (C) 2026 Stockless Variants contributors
SPDX-License-Identifier: AGPL-3.0-only
"""
from pathlib import Path


ACCESSOR = r'''
  // Partner-board search may borrow pieces from empty reserves. In particular,
  // checking drops use counts -1/-2, and custom two-board pieces need not share
  // orthodox borrowing limits. Preserve the original fast table for ordinary
  // counts; encode every other signed count without indexing outside its row.
  inline Key pocket_key(Piece pc, int count) noexcept {
      if (static_cast<unsigned>(pc) < PIECE_NB
          && static_cast<unsigned>(count) < SQUARE_NB)
          return inHand[pc][count];

      // A bijective 64-bit mixing function over (piece, signed-count bits).
      // This keeps distinct debts distinct, is deterministic across threads,
      // and does not consume PRNG values or change any ordinary position keys.
      std::uint64_t value = (std::uint64_t(static_cast<unsigned>(pc)) << 32)
                          | std::uint32_t(count);
      value += UINT64_C(0x9e3779b97f4a7c15);
      value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
      value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
      return value ^ (value >> 31);
  }
'''


def bracket(source: str, offset: int) -> tuple[str, int]:
    if source[offset] != "[":
        raise RuntimeError("pocket key patch: expected array subscript")
    depth = 1
    end = offset + 1
    while end < len(source) and depth:
        if source[end] == "[":
            depth += 1
        elif source[end] == "]":
            depth -= 1
        end += 1
    if depth:
        raise RuntimeError("pocket key patch: unterminated array subscript")
    return source[offset + 1:end - 1], end


def apply(out: Path) -> None:
    path = Path(out) / "src/position.cpp"
    source = path.read_text(encoding="utf-8")
    marker = "Zobrist::inHand"
    offset = 0
    edits: list[tuple[int, int, str]] = []
    writes = 0
    while True:
        begin = source.find(marker, offset)
        if begin < 0:
            break
        piece, after_piece = bracket(source, begin + len(marker))
        count, end = bracket(source, after_piece)
        offset = end
        if source[end:].lstrip().startswith("= rng.rand<Key>();"):
            writes += 1
            continue
        edits.append((begin, end, f"Zobrist::pocket_key({piece}, {count})"))
    # Checked against the pinned FSF revision after gating/free-drop fixes.
    if len(edits) != 13 or writes != 1:
        raise RuntimeError(f"pocket key patch: expected 13 reads and one initialization, got {len(edits)}/{writes}")
    for begin, end, replacement in reversed(edits):
        source = source[:begin] + replacement + source[end:]
    anchor = "  Key inHand[PIECE_NB][SQUARE_NB];\n"
    if source.count(anchor) != 1:
        raise RuntimeError("pocket key patch: table declaration changed")
    source = source.replace(anchor, anchor + ACCESSOR, 1)
    path.write_text(source, encoding="utf-8")
