# Efficient NNUE redesign: Stockfish + Fairy-Stockfish → Stockless Variants

Research and implementation evidence for the separate
[experimental/svnnue-v3-efficient](https://github.com/gaong5247-cmd/Stockless-variants/tree/experimental/svnnue-v3-efficient)
branch. This does NOT replace the project's classical default evaluation or
the stable v1/v2 model format. New features have separate format magic,
version, rule/feature-source digest and payload checksum.

## 1. Primary source audit

- Stockfish's maintained [NNUE internals guide](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html):
  sparse mostly-binary piece inputs, HalfKP/HalfKA king-relative features,
  accumulators updated by adding/subtracting changed input columns,
  quantized low-latency SIMD inference, factorized training features and
  WDL/game-result mixing as optional training objectives.
- [Current Stockfish nnue-pytorch trainer](https://github.com/official-stockfish/nnue-pytorch):
  large streamed/chunked .binpack inputs, sparse CPU/CUDA data kernels,
  optimizer checkpoints, independent validation and high-volume game data.
  The full upstream large-data recipe requires orders of magnitude more
  hardware and storage than the GitHub CPU smoke runner.
- Fairy-Stockfish's [variant NNUE technical details](https://github-wiki-see.page/m/fairy-stockfish/variant-nnue-pytorch/wiki/Technical-details)
  and [trainer](https://github.com/fairy-stockfish/variant-nnue-pytorch):
  HalfKAv2-style per-king features generalized to non-8x8 geometry,
  arbitrary type counts, pieces in hand, and special king constraints.
  Its [native data generator](https://github.com/fairy-stockfish/variant-nnue-tools)
  is an independent strong baseline rather than code that our SVNNUE
  format pretends to match.

Stockfish/FSF's actual nets are NOT compatible with our independent
SVNNUE-v1/v2/v3 binary format. Their sources are references, not network
weights reused without proper training and format conversion.

## 2. Diagnosis of the original SVNNUE problems

The first single-accumulator 64-unit network learned fixed piece-square
placement, reserve counts and later promoted-origin/region/rights inputs.
Across **1,247,386 generated native positions** (v1) and a separate
**1,241,675 native v2 positions**, best validation checkpoints appeared
after roughly 1–2 epochs. Paired Classical matches showed the networks
were generally weaker. This indicates incomplete representation and
generalization despite successful numerical training and large input counts.
More identical teacher data and more epochs are not sufficient evidence of
strength improvement.

The early CPU streaming trainer built dense (batch_size × input_dim)
matrices and performed dense GEMM even when most inputs were zero.
Additional v2 castling/en-passant features were extracted via
expensive FEN string formatting at every network probe.

## 3. Implemented v3 research architecture

**Not a full copy of modern Stockfish HalfKA.** The v3 model is an
independent compact 64-hidden-unit transformer with optional neural backend.
It preserves incrementally updated base board/hand features and the
v2 promotion-origin, castling, en-passant and regional features, with a
new **king-relative factor block**:

For every on-board native piece, for each valid royal perspective
(WHITE king/general, BLACK king/general), map the signed
(file-relative-to-king, rank-relative-to-king) distance into a 4×4 coarse
grid. Add one type-and-piece-color count into that perspective's
appropriate feature. Two viewpoints × two piece colors × T native
piece types × 16 distance cells = **64T extra sparse features**.

For W×H board area A=W×H and type count T:

    SVNNUE-v1: D1 = 2*T*(A+1)
    SVNNUE-v2: D2 = D1 + 2*T*A + 4 + W + 6*T
    SVNNUE-v3: D3 = D2 + 64*T

Example input dimensions:

| Game | Native type count | v1 dim | v2 dim | v3 dim |
| --- | ---: | ---: | ---: | ---: |
| Chess 8×8 | 6 | 780 | 1,596 | **1,980** |
| Shogi 9×9 | 10 | 1,640 | 3,333 | **3,973** |
| Janggi/Xiangqi 9×10 | 7 | 1,274 | 2,589 | **3,037** |
| Makruk 8×8 | 6 | 780 | 1,596 | **1,980** |

The extra royal-relation fields are recomputed from native C++ Position
during the forward probe, while base board/hand accumulator mutations remain
incremental. V3 currently has **one shared 64-wide accumulator and two
side-to-move output heads** — not Stockfish's genuine dual perspective
accumulator stack. This deliberate compromise keeps compact networks,
simple make/unmake correctness and low RAM usage; a further incremental
royal-relative cache and independent dual perspective transformers are
separate performance work.

The v3 chess castling/en-passant block uses native castling rights and
native en-passant bitboards, **not FEN serialization on each evaluation**.
Inputs still omit some history/repetition/counters. All legal moves,
terminal results and adjudications remain the original native engine's job.

Source: src/stockless_variants/nnue.cpp;
v3 UCI export: nnuefeaturesv3;
file magic: SVNNUE3 + NUL; binary header format/feature version: 3.

## 4. CPU training improvements: CSR instead of dense inputs

tools/nnue/sparse_batches.py uses SciPy's compressed sparse row (CSR)
representation, so a batch expands only its nonzero native features.

The first transformer now computes:

    H = clipped_relu(X_sparse @ W1 + b1)

and the exact dense first-layer backprop gradient is calculated by

    grad_W1 = X_sparse.T @ grad_H

Adam state, all weights, moments, optimizer step, source fingerprint,
seed, batch_size, kernel selection and best-validation checkpoint remain
checksum-bound and safely resumable. The v3 runner refuses a partial or
incompatible optimizer state.

Five independently hosted CPU tests compared CSR against dense reference
for identical 256-row, 48-feature/position, 64-hidden-unit batches across
all five games. Float/integer results and first-layer gradients matched
their reference tolerances. **Observed isolated transformer forward+
gradient speedup ranged about 4.4×–6.7×** on single-BLAS-thread runners.
This is an **operator benchmark**, not a proof that full trainer wall-clock
time or engine playing NPS rose by that factor. CPU/OS and density matter.

Dependencies are pinned separately: NumPy 2.3.5 and SciPy 1.17.0.
This is CPU-first; there is no unsupported GPU/CUDA promise.

## 5. Large-data reuse without fake new search results

The [native reencoder](../tools/nnue/reencode_native_v3.py) accepts original
checksummed, native-v2, depth-4-labeled JSONL game shards. It replays every
full legal game history in the exact v3 engine, verifies original FEN and
side-to-move, computes true v3 native sparse features, and **preserves the
original classical-teacher labels and terminal results**.

A transformed data row has distinct original teacher/source-rule SHA-256
and v3 native feature-exporter SHA-256/descriptor. It is labelled
"re-encoded" — never newly generated or newly searched. It refuses
mismatched source hashes, provenance, FEN or turn states.

Completed source [v2 depth-4 game/label campaign](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40329751018):
**1,241,675 real teacher-labeled positions**. Separate v3
[full reencoder + sparse training workflow](../.github/workflows/nnue-v3-million-efficient.yml)
uses all ten source shards, 24 CPU training epochs per game with genuine
fourth-epoch Adam optimizer checkpoint resume, and exact heldout native
inference/search verification. This is still *classical teacher
distillation*, not RL, and needs later strength testing.

## 6. Acceptance gates before any release

1. C++ build on all intended architectures; v1/v2 regression unchanged.
2. Five-variant v3 export, quantization, integer parity and legal searches.
3. Distinct game-level train/validation/test groups, no identical model input
   appearing across partitions; all input/checkpoint SHA identities match.
4. Real training corpus, heldout performance, trainer wall-clock and
   memory/engine NPS separately measured on pinned hardware.
5. Classical-v3 color-swapped paired games on fair depth/nodes/time controls,
   with incomplete games censored and confidence intervals. A small win
   fraction is not a trustworthy Elo rating.
6. No network becomes the default unless it beats Classical consistently,
   remains fast and passes sanitizer/physical Android tests.

Even a numerically perfect NNUE may be a worse evaluator if its training
labels are shallow, input representation is incomplete or it overfits
the teacher. That limitation is recorded instead of hidden.
