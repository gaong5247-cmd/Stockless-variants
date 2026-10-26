#!/usr/bin/env python3
"""Check that native v2 states encode real legal rights without corrupting v1."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"/"nnue"))
from generate import Engine, validate_state


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--engine",type=Path,required=True)
    args=parser.parse_args()
    e=Engine(args.engine.resolve(),60)
    try:
        e.option("UCI_Variant","chess")
        e.feature_version=1
        e.position([])
        old=e.features()
        validate_state(old,"chess")
        e.feature_version=2
        full=e.features()
        validate_state(full,"chess")
        d=full["descriptor"]
        types=len(d["type_ids"])
        width,height=d["width"],d["height"]
        area=width*height
        original_dim=2*types*(area+1)
        castling_offset=original_dim+2*types*area
        assert full["descriptor"]["format_version"]==2
        assert old["descriptor"]["format_version"]==1
        assert d["input_dim"]==original_dim+2*types*area+4+width+6*types
        assert [entry for entry in full["features"] if entry[0]<original_dim]==old["features"]
        active={idx for idx,val in full["features"] if val>0}
        assert all(castling_offset+i in active for i in range(4)), "Starting rights missing"
        # Legal orthodox rook move removes the corresponding native castling
        # right without any extra moving-piece or FEN reinterpretation in Python.
        e.position(["g1f3","g8f6","h1g1"])
        changed=e.features()
        validate_state(changed,"chess",d)
        after={idx for idx,val in changed["features"] if val>0}
        assert castling_offset not in after, "Stale white kingside castling right"
        assert all(castling_offset+i in after for i in (1,2,3)), "Wrong rights removed"
        assert "K" not in changed["fen"].split()[2], "Native FEN did not lose white K right"
        print("PASS native v2 extra features: v1 compatibility and castling rights on legal rook move")
    finally:
        e.close()


if __name__=="__main__":
    main()
