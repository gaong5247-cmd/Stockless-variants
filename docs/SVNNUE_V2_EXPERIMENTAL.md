# Native SVNNUE-v2 — EXPERIMENTAL BRANCH ONLY

Branch: https://github.com/gaong5247-cmd/Stockless-variants/tree/experimental/svnnue-v2

The original stable 0.2.0-dev main branch is unchanged by this experiment.
v2 uses a new file magic SVNNUE2 + NUL, format/feature version 2 and a
strictly increased input dimension. Do not rename v1 files or use v2 models
with an older executable. The conservative model fingerprint also changes
because this branch contains different engine/feature code.

Native v2 network features combine the original board and reserve counters
with:
- Promoted-piece original type on each promoted square (Shogi etc.);
- Four castling-rights flags and one en-passant file flag for Chess;
- Palace occupancy by native type and side in Janggi and Xiangqi;
- River-crossing piece counts for Xiangqi;
- Promotion-zone piece counts for Shogi and Makruk.

The extra features are computed from native C++ Position, both during
training export and at each neural evaluation. They are deliberately NOT
treated as ordinary put-piece or move-piece deltas: native promotions and
rights can change independently of a board delta. Original board/pocket
features retain incremental accumulators. V2 is slower and research-only
until evaluated and optimized.

Let A = width times height and T = native piece-type count.
- V1 board/reserve base = 2*T*(A+1)
- Original promotion type = 2*T*A
- Four castling flags = 4
- En-passant file flags = width
- Regional counters = 6*T
- V2 input dimension = base + 2*T*A + 4 + width + 6*T

These indices are exported by the native nnuefeaturesv2 UCI command.
Python does not reinterpret FEN board geometry or piece letters to create
the v2 input. The model header's format/feature version, descriptor, rule
source fingerprint, SHA-256 and quantization limits must all match.

To generate and fit experimental v2 training data:

    python tools/nnue/generate.py --nnue-version 2 --rich-context \
      --engine build/engine/src/stockless-variants --variant janggi \
      --games 32 --depth 3 --max-plies 120 --seed 20261029 \
      --output build/v2-data
    python tools/nnue/train.py --data build/v2-data \
      --output build/v2-model --epochs 20 --seed 20261029
    python tools/nnue/validate.py \
      --engine build/engine/src/stockless-variants --data build/v2-data \
      --model-dir build/v2-model --report build/v2-validation.json

The experimental CI workflow
https://github.com/gaong5247-cmd/Stockless-variants/actions/workflows/nnue-v2-experimental.yml
runs separate small training jobs for all five native games, checks v1/v2
serialization, model loading, heldout exact integer inference and
position make/unmake accumulator behavior.

Passing the pilot means only that the experimental format runs correctly.
It is NOT a claim of Elo improvement. Full million-position v1 data cannot
be silently reused as v2 input, because promotion lineage and rule rights
were not present as trained neural inputs in that dataset.
