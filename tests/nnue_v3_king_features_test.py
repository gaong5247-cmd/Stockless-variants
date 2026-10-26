#!/usr/bin/env python3
"""Verify SVNNUE-v3 schema and the native dual-king-relative feature block."""
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"/"nnue"))
from generate import Engine, validate_state
from model import quantize, write_network, read_network, integer_predict, MAGIC_V3

GAMES={
    "chess":(8,8,[1,2,3,4,5,63]),
    "shogi":(9,9,[3,4,9,11,19,20,21,22,23,63]),
    "janggi":(9,10,[4,28,29,30,32,34,63]),
    "xiangqi":(9,10,[4,6,27,29,30,31,63]),
    "makruk":(8,8,[1,2,4,6,9,63]),
}


def run_contract():
    rng=np.random.default_rng(129)
    with TemporaryDirectory() as td:
        for variant,(width,height,types) in GAMES.items():
            count=len(types)
            area=width*height
            base=2*count*(area+1)
            expanded=base+2*count*area+4+width+6*count
            d={"supported":True,"variant":variant,"format_version":3,
               "feature_version":3,"hidden_dim":64,"input_dim":expanded+64*count,
               "width":width,"height":height,"type_ids":types,
               "qa":256,"qb":64,"rules_sha256":"d"*64}
            w={"w1":rng.normal(0,.012,(d["input_dim"],64)).astype(np.float32),
               "b1":np.ones(64,dtype=np.float32)*.25,
               "w2":rng.normal(0,5,(2,64)).astype(np.float32),
               "b2":np.zeros(2,dtype=np.float32)}
            q=quantize(w,d)
            out=Path(td)/(variant+".svnnue")
            write_network(out,d,q)
            assert out.read_bytes()[:8]==MAGIC_V3
            loaded=read_network(out,d)
            x=np.zeros((2,d["input_dim"]),dtype=np.int64)
            x[0,0]=1
            x[0,expanded+13]=2
            x[1,expanded+21]=1
            sides=np.array([0,1],dtype=np.int64)
            np.testing.assert_array_equal(integer_predict(q,x,sides),
                                          integer_predict(loaded,x,sides))
            wrong=dict(d,format_version=2,feature_version=2)
            try:
                read_network(out,wrong)
                raise AssertionError("Accepted v3 file with v2 descriptor")
            except ValueError:
                pass
            print("PASS model",variant,"v3 expanded dimensions",d["input_dim"])


def run_native(path):
    e=Engine(path.resolve(),60)
    try:
        e.option("UCI_Variant","chess")
        fen="4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"
        e.send("position fen "+fen)
        e.feature_version=2
        old=e.features()
        validate_state(old,"chess")
        e.feature_version=3
        before=e.features()
        validate_state(before,"chess")
        assert before["descriptor"]["input_dim"]==old["descriptor"]["input_dim"]+64*len(old["descriptor"]["type_ids"])
        expected=[entry for entry in before["features"] if entry[0]<old["descriptor"]["input_dim"]]
        assert expected==old["features"],"V3 changed the v2 base features"
        kin=old["descriptor"]["input_dim"]
        factor_before=[entry for entry in before["features"] if entry[0]>=kin]
        assert factor_before,"V3 king-relative block was empty with both kings on board"
        assert "e1f1" in before["legal_moves"]
        e.send("position fen "+fen+" moves e1f1")
        moved=e.features()
        assert moved["descriptor"]==before["descriptor"]
        factor_after=[entry for entry in moved["features"] if entry[0]>=kin]
        assert factor_after!=factor_before,"King moved without updating king-relative features"
        print("PASS actual native v3 relative-king factors change on legal king move")
    finally:
        e.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--engine",type=Path,required=True)
    args=p.parse_args()
    run_contract()
    run_native(args.engine)


if __name__=="__main__":
    main()
