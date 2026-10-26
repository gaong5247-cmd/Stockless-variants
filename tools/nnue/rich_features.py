#!/usr/bin/env python3
"""Native-state-derived, rule-specific metadata for future SVNNUE-v2 features.

These audited diagnostics intentionally do NOT silently alter the deployed
SVNNUE-v1 64-hidden-unit input layout. A v2 runtime must be validated first.
SPDX-License-Identifier: AGPL-3.0-only
"""
from __future__ import annotations
from collections import Counter


def feature_context(state, variant):
    if state.get("schema")!=1 or "pieces" not in state or "pockets" not in state:
        raise ValueError("Rich feature extraction requires native nnuefeatures state")
    pieces=state["pieces"]
    pockets=state["pockets"]
    board=[p for p in pieces if isinstance(p,dict)]
    pocket_mass=sum(p["count"] for p in pockets)
    promoted=sum(bool(p["promoted"]) for p in board)
    promoted_origins=Counter(str(p["unpromoted_type"]) for p in board if p["promoted"])
    width,height=state["descriptor"]["width"],state["descriptor"]["height"]
    output={
        "schema":2,"variant":variant,"piece_count":len(board),"pocket_mass":pocket_mass,
        "promoted_count":promoted,"promoted_origins":dict(promoted_origins),
        "legal_mobility":len(state["legal_moves"]),
        "phase_fraction":len(board)/max(1,width*height),
    }
    if variant=="chess":
        fen=state["fen"].split()
        output["castling_rights"]=fen[2] if len(fen)>2 else "-"
        output["en_passant_target"]=fen[3] if len(fen)>3 else "-"
    elif variant=="shogi":
        output["promotion_region_occupancy"]={
           str(color):sum(1 for p in board if p["color"]==color and
                          ((p["rank"]>=height-3) if color==0 else (p["rank"]<=2)))
           for color in (0,1)}
        output["pocket_types"]=len(pockets)
    elif variant=="janggi":
        output["palace_presence"]={
            str(color):sum(1 for p in board if p["color"]==color and
                     3<=p["file"]<=5 and (p["rank"]<=2 or p["rank"]>=height-3))
            for color in (0,1)}
        output["pass_legal"]=any(m.lower() in ("0000","pass") for m in state["legal_moves"])
    elif variant=="xiangqi":
        output["river_crossed_pieces"]={
            str(color):sum(1 for p in board if p["color"]==color and
                      ((p["rank"]>=5) if color==0 else (p["rank"]<=4)))
            for color in (0,1)}
        output["palace_presence"]={
            str(color):sum(1 for p in board if p["color"]==color and
                     3<=p["file"]<=5 and (p["rank"]<=2 or p["rank"]>=height-3))
            for color in (0,1)}
    elif variant=="makruk":
        output["promotion_zone_occupancy"]={
            str(color):sum(1 for p in board if p["color"]==color and
                      ((p["rank"]>=height-3) if color==0 else (p["rank"]<=2)))
            for color in (0,1)}
    else:
        raise ValueError("Unsupported rich context variant")
    return output
