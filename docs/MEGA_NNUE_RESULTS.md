# Stockless Variants: verified 1,247,386 native NNUE training positions

All figures are from actual completed GitHub Actions jobs on 2026-10-26.
This is **supervised distillation of a depth-4 Classical evaluator**,
not NNUE playing-strength certification, self-play reinforcement learning,
or a Stockfish/FSF network claim.

## Linked evidence

- [Successful actual self-play/training run 40314151101](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40314151101): all ten generation jobs, five model jobs and final total gate passed (17 jobs including preparation).
- [Exact source revision 13ee01c4](https://github.com/gaong5247-cmd/Stockless-variants/commit/13ee01c4cca8618c84a753134a3410bdcb0954a0).
- [Separate checksum-verified long-term data/model research prerelease](https://github.com/gaong5247-cmd/Stockless-variants/releases/tag/nnue-corpus-1.247m-20261026), with ten original dataset ZIPs, five model/Adam-checkpoint ZIPs, SHA256SUMS.txt, and EXPERIMENT_MANIFEST.json (17 release assets).
- [Preregistered NNUE/Classical arena workflow and raw results](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40318979116): five successful paired-match jobs and successful aggregate job.
- [Archive validation workflow, completed successfully](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40319840423).
- [Previous 281,862-position experiment for context](LONG_TRAINING_RESULTS.md).

## Actual native data and 24-epoch training

One fixed native teacher executable and same rules source produced two independently seeded, 1,024-game, depth-4 datasets per variant. A game stopped at the 160-ply limit retains an unknown result, not a manufactured draw. Every dataset has complete move lists, native sparse features, teacher score metadata and a separate native-context-v2 diagnostic object that records selected extra rule state.

Models were trained on SQLite-indexed data using batches of 256, with whole-game train/validation/test partitioning and exact-input deduplication. Every target ran epoch 1–4, saved full Adam weights, first/second moments and step count, then resumed in a new process and completed epoch 24. Only the checkpoint with the lowest validation MSE was exported as SVNNUE-v1. The extra rich context is still not a model input feature.

| Game | Native games | Generated labeled positions | Retained unique NNUE inputs | Optimizer steps | Best validation epoch | Test teacher MAE (engine cp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chess | 2,048 | 228,135 | 225,572 | 15,312 | 2 | 264.226 |
| Shogi | 2,048 | 242,234 | 242,089 | 15,384 | 2 | 315.316 |
| Janggi | 2,048 | 241,860 | 238,791 | 15,864 | 2 | 218.533 |
| Xiangqi | 2,048 | 238,665 | 234,314 | 15,384 | 1 | 180.623 |
| Makruk | 2,048 | 296,492 | 291,664 | 18,624 | 2 | 169.993 |
| **Total** | **10,240** | **1,247,386** | **1,232,430** | **80,568** | — | Not comparable across games |

All five completed model quantization successfully, passed 128 frozen heldout C++ integer-evaluation/feature/accumulator comparisons and four depth-3 legal searches each: **660 native checks, zero failures**.

The small best-epoch values are an important **overfitting/generalization warning**. Running 24 epochs does not mean that the 24th checkpoint was strongest. Test MAE is measured against that game's own shallow teacher and is neither an Elo metric nor a direct strength comparison with the previous experiment's different heldout sets.

## Measured NNUE vs Classical games (not guesses)

Five real UCI native engine match jobs used the exact checksummed new SVNNUE weights, not the old 281k corpus models. For each game: 24 independently seeded openings, each played twice with NNUE swapping color; search depth 3, maximum 240 plies and native game-result adjudication. Every move was checked for legality and NNUE mode verified active.

| Game | NNUE wins | Draws | Losses | Unfinished at ply limit | Score, completed games only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Chess | 4 | 1 | 43 | 0 | 9.38% |
| Shogi | 2 | 1 | 45 | 0 | 5.21% |
| Janggi | 13 | 0 | 35 | 0 | 27.08% |
| Xiangqi | 9 | 10 | 29 | 0 | 29.17% |
| Makruk | 1 | 4 | 36 | 7 | 7.32% |
| **Total** | **29** | **16** | **188** | **7** | Do not pool variants for Elo |

**Classical remains decisively stronger** in this bounded comparison. The pilot changes the strength-evidence picture relative to the early network experiments, but the openings and sample sizes differ, so the figures do not prove a particular Elo improvement over previous networks. Seven uncompleted Makruk games remain censored, never counted as draws.

[Raw match history for all 240 games](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40318979116).

## Higher-depth 480-game cross-check (independent follow-up)

The [successful depth-4 arena run 40319511867](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40319511867)
is a *second and distinct* actual experiment. It used the exact same
million-position models with matching SHA-256 network identities, but moved
to **depth 4**, **48 paired openings** per variant (96 games per variant),
a 300-ply cutoff and a new deterministic opening seed (20261027). Its
native-turn-based player assignment uses the actual engine's
`side_to_move` rather than assuming every game starts with the same color.
Each color-swapped opening pair was checked by the aggregate job.
All five match jobs and the aggregate job passed.

| Game | NNUE wins | Draws | NNUE losses | Unfinished | Completed-only score | Game histories |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Chess | 4 | 3 | 88 | 1 | 5.79% | [JSON](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40319511867/artifacts/12648007623) |
| Shogi | 9 | 0 | 86 | 1 | 9.47% | [JSON](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40319511867/artifacts/12648406844) |
| Janggi | 23 | 0 | 73 | 0 | 23.96% | [JSON](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40319511867/artifacts/12647493838) |
| Xiangqi | 24 | 14 | 56 | 2 | 32.98% | [JSON](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40319511867/artifacts/12647744342) |
| Makruk | 1 | 6 | 86 | 3 | 4.30% | [JSON](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40319511867/artifacts/12647957935) |
| **Total** | **61** | **23** | **389** | **7** | **Do not pool variants for Elo** | [Aggregate](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40319511867/artifacts/12648791966) |

All incomplete games remain **censored**, not draws. The per-game
percentages use only completed matches. Although sample sizes increased,
this is still not an Elo estimate: different rules and color/opening
samples cannot be pooled as if one homogeneous rating pool. Depth-3
and depth-4 scores are *not a controlled network-only comparison*, as the
search depth and independent opening sample also changed. Classical
remains stronger on all five games. Do not enable these experimental
models as the default evaluator based on these results.

## Download and reproduce

The separate research [Release archive](https://github.com/gaong5247-cmd/Stockless-variants/releases/tag/nnue-corpus-1.247m-20261026) contains exactly 17 files: ten native JSONL data archives, five model/optimizer archives, a machine-readable experiment manifest and SHA256SUMS.txt. It is a prerelease, **not the latest engine release**. The existing latest engine release remains [v0.2.0-dev](https://github.com/gaong5247-cmd/Stockless-variants/releases/tag/v0.2.0-dev). Release assets persist outside short-lived Actions artifacts, subject to repository/release retention.

Use a correctly compiled compatible 0.2.0-dev Stockless Variants binary, choose the matching game in UCI, load that game's SVNNUE file with EvalFile, and explicitly select EvalBackend=StocklessNNUE. Do not feed these custom .svnnue files to Stockfish/FSF NNUE loaders. Neural evaluation is disabled by default and should remain optional while strength results are poor.

## Next engineering requirements

Actual **SVNNUE-v2 rule-aware neural features and corresponding native inference** are still needed; current extra game-specific fields are only recorded diagnostics and are not consumed by the v1 model. Also needed are stronger multi-depth teachers, better independent positions, long matched time-control games, and improved generalization before any model promotion. A complete sample of million-plus real data proves scale and provenance, not chess-engine quality.
