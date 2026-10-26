#!/usr/bin/env python3
"""Checked fixes for reproducible state/hash defects in the pinned FSF core.

These do not alter move generation. They repair the Ataxx free-drop pocket
hash access, Seirawan pocket/material incremental hashes, and omitted gating
rights in the transposition/repetition key. Regression coverage is in
tests/rules_test.cpp. Upstream source is never edited in place.
"""
from pathlib import Path
import re


def apply(out: Path) -> None:
    path = out / "src/position.cpp"
    text = path.read_text(encoding="utf-8")

    def replace(old: str, new: str) -> None:
        nonlocal text
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f"position.cpp rule patch expected one anchor, found {count}: {old[:70]!r}")
        text = text.replace(old, new, 1)

    replace("  Key wall[SQUARE_NB];", "  Key wall[SQUARE_NB];\n  Key gate[COLOR_NB][SQUARE_NB];")
    anchor = "  chess960 = isChess960 || v->chess960;\n"
    replace(anchor, """  // Castling letters may imply gates while parsing a Seirawan FEN even
  // after its reserve is exhausted. Match do_move's canonical zero gates;
  // castling rights remain independent, and no future reserve can appear.
  if (seirawan_gating() && !captures_to_hand())
      for (Color c : {WHITE, BLACK})
          if (count_in_hand(c, ALL_PIECES) == 0)
              st->gatesBB[c] = 0;

""" + anchor)
    anchor = "  // Prepare the cuckoo tables\n"
    replace(anchor, """  // Gating eligibility affects legal moves and must distinguish TT/repetition keys.
  for (Color c : {WHITE, BLACK})
      for (Square s = SQ_A1; s <= SQ_MAX; ++s)
          Zobrist::gate[c][s] = rng.rand<Key>();

""" + anchor)
    anchor = "  si->key ^= Zobrist::castling[si->castlingRights];\n"
    replace(anchor, anchor + """
  if (gating())
      for (Color c : {WHITE, BLACK})
          for (Bitboard gates = si->gatesBB[c]; gates;)
              si->key ^= Zobrist::gate[c][pop_lsb(gates)];
""")
    replace("""      k ^=  Zobrist::psq[pc][to]
          ^ Zobrist::inHand[pc_hand][pieceCountInHand[color_of(pc_hand)][type_of(pc_hand)] - 1]
          ^ Zobrist::inHand[pc_hand][pieceCountInHand[color_of(pc_hand)][type_of(pc_hand)]];
""", """      k ^= Zobrist::psq[pc][to];
      // Free-drop variants do not consume a pocket piece: count-1 would be -1.
      if (!free_drops())
          k ^= Zobrist::inHand[pc_hand][pieceCountInHand[color_of(pc_hand)][type_of(pc_hand)] - 1]
             ^ Zobrist::inHand[pc_hand][pieceCountInHand[color_of(pc_hand)][type_of(pc_hand)]];
""")
    replace("""      return k ^ Zobrist::psq[pc][to] ^ Zobrist::inHand[pc_hand][pieceCountInHand[color_of(pc_hand)][type_of(pc_hand)]]
            ^ Zobrist::inHand[pc_hand][pieceCountInHand[color_of(pc_hand)][type_of(pc_hand)] - 1];
""", """      // Speculative prefetch must obey the same free-drop pocket contract.
      k ^= Zobrist::psq[pc][to];
      if (!free_drops())
          k ^= Zobrist::inHand[pc_hand][pieceCountInHand[color_of(pc_hand)][type_of(pc_hand)]]
             ^ Zobrist::inHand[pc_hand][pieceCountInHand[color_of(pc_hand)][type_of(pc_hand)] - 1];
      return k;
""")
    replace("""      if (captures_to_hand())
      {
          Piece removeFromHand =""", """      if (captures_to_hand() && !free_drops())
      {
          Piece removeFromHand =""")
    # Both ordinary captures and explosion bycatch can put pieces in hand.
    capture_hash = re.compile(r"(?m)^( +)k \^=  Zobrist::inHand\[pieceToHand\]\[pieceCountInHand\[color_of\(pieceToHand\)\]\[type_of\(pieceToHand\)\] - 1\]\n +\^ Zobrist::inHand\[pieceToHand\]\[pieceCountInHand\[color_of\(pieceToHand\)\]\[type_of\(pieceToHand\)\]\];")
    def guarded_capture(match: re.Match) -> str:
        indent = match.group(1)
        return (indent + "if (!free_drops())\n" + indent + "    k ^= Zobrist::inHand[pieceToHand][pieceCountInHand[color_of(pieceToHand)][type_of(pieceToHand)] - 1]\n"
                + indent + "       ^ Zobrist::inHand[pieceToHand][pieceCountInHand[color_of(pieceToHand)][type_of(pieceToHand)]];")
    text, count = capture_hash.subn(guarded_capture, text)
    if count != 2:
        raise RuntimeError(f"expected two capture-pocket hash anchors, found {count}")
    replace("""      k ^= Zobrist::psq[gating_piece][gate];
      st->materialKey ^= Zobrist::psq[gating_piece][pieceCount[gating_piece]];
      st->nonPawnMaterial[us] += PieceValue[MG][gating_piece];
""", """      k ^= Zobrist::psq[gating_piece][gate];
      if (!free_drops())
          k ^= Zobrist::inHand[gating_piece][pieceCountInHand[us][gating_type(m)] + 1]
             ^ Zobrist::inHand[gating_piece][pieceCountInHand[us][gating_type(m)]];
      // put_piece has already incremented the piece count.
      st->materialKey ^= Zobrist::psq[gating_piece][pieceCount[gating_piece] - 1];
      if (type_of(gating_piece) == PAWN)
          st->pawnKey ^= Zobrist::psq[gating_piece][gate];
      else
          st->nonPawnMaterial[us] += PieceValue[MG][gating_piece];
""")
    anchor = "  // Update the key with the final value\n  st->key = k;"
    replace(anchor, """  // Only changed gate squares affect the incremental key.
  if (gating())
      for (Color c : {WHITE, BLACK})
          for (Bitboard changed = st->previous->gatesBB[c] ^ st->gatesBB[c]; changed;)
              k ^= Zobrist::gate[c][pop_lsb(changed)];

""" + anchor)
    path.write_text(text, encoding="utf-8")
