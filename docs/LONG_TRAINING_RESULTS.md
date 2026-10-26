# Large NNUE experiment — verified run 40309268117

**Completed successfully on 2026-10-26, 10:46:19 UTC (19:46:19 KST).**
This is an actual, completed GitHub Actions experiment, not a planned run.

- [Full workflow run and job logs](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117)
- [Source commit for the entire run](https://github.com/gaong5247-cmd/Stockless-variants/commit/2111db87ce62446b932e55f87a79198f1d09f3c5)
- [Workflow implementation](../.github/workflows/train-long-nnue.yml)
- [Shard merger and safety checks](../tools/long_training/merge.py)
- [First 0.2.0-dev baseline experiment](NNUE_RESULTS.md)

## Actual generation

Ten checksum-bound native classical-teacher shards completed, each containing
256 independent generated games. Every model combined exactly two distinct
shards from an identical teacher executable. **2,560 games and 281,862
labeled positions** were generated. No position count is an estimate.

| Variant | Shard 0 positions | Shard 1 positions | Total positions | Games |
| --- | ---: | ---: | ---: | ---: |
| Chess | 26,064 | 26,415 | **52,479** | 512 |
| Shogi | 28,301 | 28,314 | **56,615** | 512 |
| Janggi | 27,218 | 27,163 | **54,381** | 512 |
| Xiangqi | 27,575 | 26,905 | **54,480** | 512 |
| Makruk | 32,046 | 31,861 | **63,907** | 512 |
| **Total** | **141,204** | **140,658** | **281,862** | **2,560** |

The bounded experimental recipe was 256 games per shard, maximum 136 plies per
game, eight random legal opening plies, classical teacher depth 4, SHA-bound
native feature export, and 60 CPU training epochs per variant. Shards differed
in generation seed by 1,000,000, preserving distinct game IDs. No partial game
at the ply limit was relabeled a draw.

### Download original datasets

The original game histories, FENs, features, labels and complete provenance
manifests are in separate GitHub Actions artifacts:

| Variant | Shard 0 | Shard 1 |
| --- | --- | --- |
| Chess | [data 0](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12643633625) | [data 1](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12642999347) |
| Shogi | [data 0](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12643174176) | [data 1](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12643782873) |
| Janggi | [data 0](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12644355593) | [data 1](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12643229351) |
| Xiangqi | [data 0](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12644730233) | [data 1](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12644101878) |
| Makruk | [data 0](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12643733223) | [data 1](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12644655981) |

The dataset artifacts have 14-day retention, with current expiry
**2026-11-09**. This repository does not permanently store the large JSONL
datasets. Download/archive them if they are needed beyond artifact retention.

## Actual combined-model training

All five model jobs completed successfully after generation and merge.
Every one of the 60 requested epochs ran. A *selected epoch* below is the
checkpoint chosen by the best validation loss, not the number of epochs run.

| Variant | Gradient steps | Selected epoch | Heldout float teacher MAE | Quantization gate | Native NNUE checks |
| --- | ---: | ---: | ---: | --- | --- |
| Chess | 8,160 | 3 | 312.725 native cp | Passed | **132 / 132** |
| Shogi | 8,880 | 6 | 354.556 native cp | Passed | **132 / 132** |
| Janggi | 8,460 | 3 | 322.282 native cp | Passed | **132 / 132** |
| Xiangqi | 8,520 | 3 | 238.410 native cp | Passed | **132 / 132** |
| Makruk | 9,900 | 2 | 183.266 native cp | Passed | **132 / 132** |
| **Total** | **43,920** | — | — | **5 / 5** | **660 / 660** |

Every job generated a different per-variant .svnnue network, quantized it
successfully, loaded it into the actual Stockless Variants engine, checked
**128 frozen heldout integer/feature/accumulator positions and 4 legal depth-3
searches**, and uploaded the model plus checkpoint, split/metrics and model
identity manifest. Native check failure count: zero.

These MAEs measure **approximation to that variant's depth-4 classical
teacher**, not playing strength, and are not comparable across variants.
The short prior pilot used depth 3, a different seed/data split and far less
data, so a direct MAE difference cannot establish strength improvement.
The early selected epochs are evidence to review overfitting/validation
generalization, not to hide in favor of the last epoch.

### Download trained models

| Model | GitHub Actions artifact |
| --- | --- |
| Chess | [long chess NNUE](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12644441027) |
| Shogi | [long shogi NNUE](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12644480804) |
| Janggi | [long janggi NNUE](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12644131890) |
| Xiangqi | [long xiangqi NNUE](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12644985577) |
| Makruk | [long makruk NNUE](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40309268117/artifacts/12644016422) |

Current model artifact expiry is **2026-11-25**. The NNUE-compatible
0.2.0-dev engine is available separately in [DOWNLOADS.md](DOWNLOADS.md).
Activate a matching model explicitly through EvalFile and
EvalBackend=StocklessNNUE. All models remain experimental and
**Classical stays enabled by default**.

## What is not yet done

- Controlled NNUE-vs-Classical matched games with confidence intervals;
  **no Elo increase has been measured**.
- Sparse streaming that would exceed the present 100,000-row and 512 MiB dense
  trainer limits, nor optimizer-state checkpoint resume across workflow runs.
- More informative features for promoted-piece lineage, castling, en-passant,
  adjudication and repetition/history; richer network architectures.
- A stronger independent teacher, multi-seed heldout robustness studies,
  or automatic promotion to a production evaluator.

This experiment completed in one sharded workflow, rather than requiring a
single GitHub runner to run longer than its job limit. The workflow can be
launched manually with different bounded parameters, but a fresh run starts
new weights unless a future verified resume protocol is implemented.
