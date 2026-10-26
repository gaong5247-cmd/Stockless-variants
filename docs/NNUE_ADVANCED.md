# Experimental streaming NNUE, resume and paired matches

Source implementation: [stream_dataset.py](../tools/nnue/stream_dataset.py),
[train_stream.py](../tools/nnue/train_stream.py),
[validate_stream.py](../tools/nnue/validate_stream.py),
[arena.py](../tools/nnue/arena.py), and
[rich_features.py](../tools/nnue/rich_features.py).

## Streaming large corpora

SQLite verifies dataset JSONL SHA-256 and byte length, game IDs, teacher/rules
identity, labels and native sparse feature ordering. Whole-game train/validation/
test splits are deterministic. The test partition is held out until the best
validation checkpoint is selected. Identical input states cannot occur across
retained partitions.

Training converts only the current bounded mini-batch into a dense NumPy array,
not all positions. The original 100k-row / 512 MiB full-tensor limit no longer
applies, but dataset and index disk storage increase with corpus size.
The synthetic stress test targets **1,000,001 position rows** to verify
disk-backed indexing and bounded process memory. Synthetic positions are not
equivalent to a million high-quality real-engine training samples.

Run:

    python tools/nnue/train_stream.py --data /data/chess \
      --output build/stream-chess --epochs 20 --batch-size 128 --seed 20261026

## Exact optimizer-state resumption

Every completed epoch saves a checksummed versioned NPZ with parameters,
Adam moments, variances, selected-best weights, plus an atomic latest.json
pointer containing epoch, step count, seed, dataset identity and batch size.
Future-total epochs do not affect the LR schedule. Bit-exact reproducibility
is tested: one epoch + resume to two equals uninterrupted two epochs,
including Adam state and the exported quantized SVNNUE file.

    python tools/nnue/train_stream.py --data /data/chess \
      --output build/stream-chess --epochs 20 --stop-after-epoch 4
    python tools/nnue/train_stream.py --data /data/chess \
      --output build/stream-chess --epochs 20 --resume

Refusing an altered dataset/seed, missing file or corrupted checkpoint is
intentional. Keep each dataset immutable for the full training run.

Native runtime validation:

    python tools/nnue/validate_stream.py \
      --engine build/engine/src/stockless-variants \
      --index build/stream-chess/index.sqlite3 \
      --model-dir build/stream-chess --positions 64 \
      --report build/stream-chess/runtime-validation.json

## Rich game-specific features — schema 2, research only

With the optional --rich-context switch, native game-state exports now record
promoted-piece origin, pocket contents, game phase and legal mobility.
Chess additionally records castling/en-passant fields, Shogi promotion regions,
Janggi palace/pass, Xiangqi river and palace occupancy, and Makruk promotion
zones. These are stored as native_context_v2 diagnostics on each generated
position. They do NOT silently change the deployed SVNNUE-v1 neural input
layout: a corresponding versioned C++ NNUE-v2 evaluator and accumulator have
not yet been implemented. Their use as trained inputs is a separate gate.

## Paired NNUE versus Classical engine games

The arena starts separate neural/classical processes of the same engine build,
requires an active compatible network, and plays two games per shared random
opening with colors reversed. Native legal moves and termination rules are
checked; positions must agree between both processes. Truncated games retain
null outcomes, never synthetic draws. Every full move history, network and
engine hash, W/D/L, draw, unfinished count and a descriptive completed-only
score interval are included. No Elo is invented.

    python tools/nnue/arena.py \
      --engine build/engine/src/stockless-variants \
      --model /weights/chess.svnnue --variant chess \
      --pairs 30 --depth 4 --max-plies 240 \
      --output build/arena-chess.json

[Advanced CI workflow](../.github/workflows/nnue-advanced.yml) runs a
deterministic resume test, native data/training/NNUE replay smoke, small
paired match series for five variants, and optional million-row load test.
Do not promote a model solely because it passed numerical parity.
