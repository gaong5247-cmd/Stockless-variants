# NNUE streaming, resume and independent paired-arena results

Recorded on 2026-10-26 (UTC), using actual completed GitHub Actions
runs. Neither teacher approximation nor quantization accuracy is playing
strength.

## One-million-unique-row memory stress

- [Completed CI run 40313573431](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40313573431)
- **1,000,001 input position rows, all 1,000,001 unique** (synthetic sparse test fixtures).
- Peak Python process resident set: **48,112 KiB (~46.98 MiB)**.
- SQLite index size: **293,560,320 bytes**.
- Dataset hashes, native feature bounds, disjoint whole-game split and row
  counts were audited. This demonstrates the indexing portion can handle more
  than one million inputs without building a giant in-RAM dense feature array;
  it is NOT a measured one-million-epoch training benchmark.
- Deterministic full-Adam checkpoint/resume comparison passed exactly,
  including optimizer moments, variance, best weights, and exported model.
- The separate native chess smoke used actual generated FEN/moves and rich
  native-context metadata, trained with one epoch + resume to epoch two, and
  passed exact native integer/accumulator and legal-search checks.

## Stream-training actual depth-4 teacher positions

- [Completed 5-variant streaming model run 40313573444](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40313573444)
- Source: [previous 2,560-game / 281,862-position campaign](LONG_TRAINING_RESULTS.md).
- Each of five models ran 2 epochs, checkpointed, restored Adam state in a
  separate process, and finished epoch six.
- Each model passed 64 heldout exact C++ integer/feature/accumulator comparisons
  and 4 depth-3 legal searches: **340/340 native checks, no failures**.

| Variant | Test teacher-approximation MAE (native cp) | Selected epoch | Native checks | Model + full optimizer state |
| --- | ---: | ---: | --- | --- |
| Chess | 310.179 | 5 | 68/68 | [artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40313573444/artifacts/12646285768) |
| Shogi | 362.437 | 5 | 68/68 | [artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40313573444/artifacts/12645009437) |
| Janggi | 335.252 | 6 | 68/68 | [artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40313573444/artifacts/12646335356) |
| Xiangqi | 258.919 | 2 | 68/68 | [artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40313573444/artifacts/12646625167) |
| Makruk | 186.464 | 2 | 68/68 | [artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40313573444/artifacts/12645418444) |

The cp scale is the corresponding native variant teacher scale, not an Elo.
Holdout subsets and model checkpoints differ from previous experiments.

## Actual matched NNUE versus Classical games

[Five-job run 40314379070](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40314379070)
ran **16 paired openings / 32 games each** using precisely the streamed
network files above against the same build's Classical backend.
Both players used the same depth-3 search setting, 240-ply game limit and
identical randomized openings with colors reversed. No censored games occurred.
The NNUE was explicitly loaded and validated as active. Native legal moves
and terminal outcomes were checked. Full moves and terminal outcomes are in
the attached per-game JSON artifacts.

| Variant | NNUE wins | Draws | NNUE losses | Recorded matches |
| --- | ---: | ---: | ---: | --- |
| Chess | 0 | 0 | 32 | [game histories](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40314379070/artifacts/12646675578) |
| Shogi | 0 | 0 | 32 | [game histories](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40314379070/artifacts/12646356739) |
| Janggi | 2 | 0 | 30 | [game histories](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40314379070/artifacts/12646221481) |
| Xiangqi | 1 | 3 | 28 | [game histories](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40314379070/artifacts/12645269880) |
| Makruk | 0 | 1 | 31 | [game histories](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40314379070/artifacts/12645602846) |

These are unambiguously poor pilot results for all five *specific* networks;
none should replace Classical as default. The sample is small, color-paired
but not a large statistical strength rating, and different variants must not
have Elo scores compared with one another.

### Rich features boundary

[Native context v2](../tools/nnue/rich_features.py) adds original promotion
identity, rights, board-region occupancy, pockets and legal mobility to *training
data diagnostics*. The currently deployed SVNNUE-v1 networks do not consume
these extra inputs. Adding those inputs to a trained network requires a
separately versioned native runtime, feature adapter, incremental update
semantics and new validation. A better training loss alone cannot compensate
for information that the NNUE input does not contain.

## Ongoing separate real million-position campaign

The [10,240-game target campaign](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40314151101)
uses native teacher positions (not synthetic stress input). Its final
summary must record exact generated count and will fail if below 1,000,000.
This document does not claim the target is complete until that run has
actually concluded. Native-context metadata is collected, but the fitted
network remains SVNNUE-v1.
