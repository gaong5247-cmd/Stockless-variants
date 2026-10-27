# SVNNUE-v3 Lite — verified native ISA parity and short NPS baseline

Research branch: [experimental/svnnue-v3-lite](https://github.com/gaong5247-cmd/Stockless-variants/tree/experimental/svnnue-v3-lite).
All results are **experimental**; the default Classical evaluator and main
0.2.0-dev release are unchanged.

## Implementation

- 16 royal-region buckets on five native games. Each player's royal anchor
  produces a separate sparse transformer perspective, followed by a shared
  128-channel feature transformer per side and a 256 -> 32 -> 16 -> 1 quantized
  ClippedReLU network.
- Whole-game train/validation/test partition and CSR minibatches; full Adam
  optimizer checkpoint at epoch 3 and resume to epoch 8.
- Incremental board/pocket additions/removals on each perspective. Any royal
  move invalidates the cached dual accumulator. Evaluation uses row-major
  contiguous tail-weight traversal, zero-activation skipping, int64 tail sums
  and an AVX2 8-lane int16-weight accumulate implementation.
- Compiled-backend information is exposed in native nnuestatus:
  accumulator_implementation = scalar or avx2.
- The experimental SVNNUE3 file uses its own versioned header and exact
  rule/source fingerprint. It is **not** a Stockfish .nnue file and must not
  be silently loaded as a v1/v2 model.

## Measured tests

[Actual completed scalar/AVX2 parity and training workflow #40470010550](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40470010550)
at exact source commit dd59c6851f2f0094fe348b284cea9981917a6ae7.

All five independently trained v3 networks passed native heldout integer
evaluation, accumulator full-refresh, legal search and quantization gates.
The subsequent parity jobs built **two distinct binaries** from the same
source commit: scalar x86-64 and x86-64-avx2. Both loaded the same trained
network and replayed the same original whole-game histories. The compiled
code path was confirmed by separate native nnuestatus values. Every
comparison checked both native feature lists, actual native FEN, signed
centipawn evaluation, cached vs fully refreshed accumulators, and post-search
root values.

**Direct parity: 305 heldout positions, 30 search/make-unmake cases,
zero scalar/AVX2 integer mismatches and zero failed jobs.** Xiangqi had
49 eligible heldout inputs; all were tested.

### Short fixed-depth throughput baseline

The same workflow measured depth-7 search on **8 identical opening sequences**
per variant, with Threads=1, Hash=16, and a fixed model. Timings are short
and the search trees can differ. This is *whole-search NPS*, not a pure NNUE
microbenchmark or an estimate of playing strength.

| Variant | Scalar v3 search NPS | AVX2 v3 search NPS | Measured AVX2 / scalar | AVX2 Classical search NPS |
| --- | ---: | ---: | ---: | ---: |
| Chess | 231,524 | 395,521 | 1.708x | 436,208 |
| Shogi | 188,279 | 331,881 | 1.763x | 323,129 |
| Janggi | 208,159 | 326,875 | 1.570x | 300,179 |
| Xiangqi | 227,914 | 395,213 | 1.734x | 374,201 |
| Makruk | 196,026 | 380,617 | 1.942x | 468,817 |

The AVX2 build changes compiler optimization for the **whole engine**,
not just the NNUE accumulator. The ratio is a build-level measurement,
not an isolated causal estimate of the SIMD routine.

Original per-variant parity and benchmark JSON archives from that run:

- [Chess](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40470010550/artifacts/12718282722)
- [Shogi](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40470010550/artifacts/12718486241)
- [Janggi](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40470010550/artifacts/12717414587)
- [Xiangqi](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40470010550/artifacts/12718586013)
- [Makruk](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40470010550/artifacts/12718263528)

## Playing strength is still an open problem

The [earlier color-swapped pilot #40465094883](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40465094883),
using small eight-epoch V3 networks, yielded NNUE 1 win, 0 draws, 118 losses,
and 1 censored game across five different variants (120 total games). Thus
**working native inference and a faster AVX2 build do not prove that v3 is
stronger than Classical**. No Elo number should be inferred from those
heterogeneous short matches. Those particular poor pilot weights remain
experimental and must not be default-enabled.

### More robust benchmark

A separate [20-position depth-9 benchmark](https://github.com/gaong5247-cmd/Stockless-variants/actions/workflows/v3-nps-depth9.yml)
has been configured. It tests the same exact trained model and openings on
both independently compiled backends, enforces complete sample coverage and
records actual nodes/time. Its results supersede the noisy short NPS
measurements when verified; until then the numbers above are preliminary.

### Next acceptance gates

Train on a large *actual v3-oriented native corpus*, not relabeled v1/v2
features; use deeper and more diverse teacher labels plus completed-game
outcomes with valid provenance. Build a broader C++ make/unmake regression
covering royal moves, drops and special moves, then run significant color-paired
NNUE-v3 vs Classical matches at a fixed hardware/time budget before changing
the engine default. Quantitative NPS and exact integer parity are necessary,
but not sufficient, for model promotion.
