# SVNNUE-v3 design review: Stockfish / Fairy-Stockfish architecture

Status: **design specification only**. Existing SVNNUE-v1 and experimental
SVNNUE-v2 remain the deployed/tested implementations. This design must pass
all gates before implementing or promoting a new network.

## Primary technical references

- Stockfish official NNUE documentation:
  https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html
- Official training code:
  https://github.com/official-stockfish/nnue-pytorch
- Fairy-Stockfish variant trainer:
  https://github.com/fairy-stockfish/variant-nnue-pytorch
- Fairy-Stockfish generation tools:
  https://github.com/fairy-stockfish/variant-nnue-tools
- FSF NNUE application and rules:
  https://github.com/fairy-stockfish/fairy-stockfish.github.io/blob/main/about-nnue.markdown

## Why v2 is still weak

The existing v1/v2 native networks have a 64-unit board feature transformer
with just two scalar output heads. They have no per-royal perspective, no
relative king/piece interaction buckets and no deep nonlinear section.
Even 1.247 million labeled positions with a depth-4 Classical teacher
do not remove the fundamental information bottleneck. The 24-epoch pilot
selected checkpoints mostly from epochs 1-2 and remained much weaker than
Classical in paired matches. This is not solely an optimizer problem.

The v2 extra state (promoted origin, rights, regional counts) is a useful
correctness improvement; its dense-backend initial implementation still
recomputes volatile features from Position on each probe and needs a
performance measurement. Increasing channels on the same simplistic
architecture without measuring NPS can make the engine worse.

## V3 proposal

### 1. Separate trainer, network format and native evaluator

Introduce a new network schema and magic distinct from SVNNUE-v1/v2.
Compile-time plus file-header checks must bind rules fingerprint, feature
schema, variant, royal semantics, network dimensions, quantization,
perspective count, and checksum. Never reinterpret legacy v1/v2 weights
as a new architecture. Preserve Classical as the default.

### 2. Two explicit perspectives

For each player-relative royal anchor, compute separate sparse features
(indexed by king/royal region or king bucket, piece type/color, square).
Concatenate transformed outputs for white and black; swap perspective order
according to side to move. For variants without a single royal king, use
a separately declared generalized orientation or anchor regime, never
fabricate a standard king square. The feature schema is variant-specific.

Use moderately sized king buckets initially to avoid very rare full
king-square-conditioned features. Factorize shared piece-square embeddings
during training so underrepresented buckets borrow statistical strength.
Do not blindly copy the much larger modern Stockfish dimensions: benchmark
our games, positions and CPU resources.

### 3. True sparse, mostly incremental input transformer

Maintain a per-position accumulator for each perspective. An ordinary
non-king move applies subtract/add deltas of the affected transformed
columns; castling, drops, promotions and unmake each get explicitly tested
update contracts. King/anchor bucket changes invalidate only the impacted
perspective and cause lazy refresh. Cache refresh predicates by state
generation and preserve search-thread ownership. Never mutate a shared
root StateInfo from multiple threads.

Volatile state features (rights, en passant, special Janggi/Xiangqi regional
counters and promoted lineage) need their own event-aware diff or a separate
small tail branch, not the expensive full dynamic scan currently used by
the experimental v2 code. Compare incremental state to from-scratch
evaluation on every move type and on make/unmake, including null moves.

### 4. Architecture search, not a single giant model

Benchmark candidates:
- V3-lite: 2 perspectives x 128 transformer channels, 32 -> 16 -> 1 tail.
- V3-balanced: 2 perspectives x 256 channels, 64 -> 32 -> 1 tail.
- V3-large: 2 perspectives x 384 channels, 64 -> 32 -> 1 tail.

Use clipped ReLU / low-precision compatible activations. Quantized int8/
int16 transformer storage and bounded int32 accumulators need independent
signed overflow proofs; retain int64 final sums. SIMD scalar/AVX2/ARM NEON
parity is mandatory. Evaluate king-bucket index sparsity and worst-case
recomputation before any large corpus training.

### 5. Training data and targets

Use native legal self-play and independent diversified positions (not only
teacher-chosen moves), deeper evaluation labels and actual completed
results where available. A truncated position retains unknown game outcome.
Repetition and adjudication histories must be native; do not infer
variant draws from insufficient material. Stratify the train/validation/
test groups by whole games AND opening families, deduplicate identical
feature/side/royal states, check that all labels use the same teacher
semantic scale, and freeze the test set.

Use a mixed target of transformed cp/regression labels and real outcome
probabilities *only for finished games*, with explicit data weights and
separate diagnostics. A depth-4 classical score alone mostly teaches the
network to imitate a weak classical teacher.

### 6. Efficient CPU/GPU training

Keep disk-backed shuffled shards and a versioned manifest. Prefer packed
native feature-index arrays and fixed size batched sparse gather/scatter,
similar to official nnue-pytorch. Existing CSR support is an improvement
over a full dense matrix but still materializes a dense first-layer gradient:
benchmark sparse fused update and native/C++ multithreaded batch loader
separately. A reader with workers should prefetch complete minibatches
asynchronously to avoid Python JSONL per-sample overhead.

GPU training should be optional, based on actual available backend and
memory, with a CPU correctness fallback. Test fp32 train vs fixed-point
engine output independently, including quantization saturation and
validation MAE/p99/max. Do not assume speedups from CUDA, OpenVINO or XPU
without measured hardware traces.

Adam moment, variance, global step, LR scheduler, data epoch/shuffle state
and best validation checkpoint must be checksum-bound for exact resume.
Archive original game histories/data, not only compressed networks.

### 7. Acceptance gates and strength metrics

A. Perft/native result equivalence, original v1/v2 regression and legal
   make/unmake/state hash checks remain green.
B. Signed accumulator exact parity on 10k real heldout positions across all
   five games and all supported SIMD backends.
C. Negative malformed and cross-variant model loader tests, promotion/
   drops/castling/null/palace/cannon/pinned-piece regression fixtures.
D. NPS at equal hardware/hash/thread/time/settings: compare Classical,
   SVNNUE-v2, and each v3 candidate on the exact same search positions.
   Reject large NPS regressions unless paired matches establish a benefit.
E. At least 1000 independent color-swapped paired actual games per variant
   for production strength decisions (pilot: 100/variant). Confidence
   intervals, opening identity, censored games, crash/time/illegal move
   counts and engine/model SHA are mandatory. Do not pool variant-specific
   Elo values or promote a model based on teacher MAE alone.
F. Only a candidate passing all rule, numerical, speed and match gates
   can become opt-in beta; Classical remains default until qualification.

## Immediate implementation milestones

1. Freeze the SVNNUE-v2 corpus and native feature schema; benchmark v1/v2
   and record low-level feature activation frequencies/king anchor coverage.
2. Implement V3-lite format/loader and perspective-aware features in an
   isolated branch, with Python and C++ bit-exact parity before training.
3. Implement king-bucket lazy accumulator refresh, SIMD and multi-threaded
   state tests; compare NPS before increasing hidden width.
4. Train V3-lite with sparse packed batches and a reproducible multi-depth
   teacher on existing native positions regenerated in the new feature
   schema; full Adam checkpointing and heldout splits.
5. Run 100-pair pilot per variant then expand matches only when results
   justify it. Publish raw games and checkpoint artifacts separately.

This design captures transferable principles, not a claim that this engine
implements Stockfish 19's entire architecture. Native Fairy-Stockfish
variant rules and Stockless hybrid search remain distinct components.
