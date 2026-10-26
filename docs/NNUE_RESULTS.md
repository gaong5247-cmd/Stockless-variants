# First workflow-trained SVNNUE models

**All five independent training jobs and all five platform build jobs completed successfully on 2026-10-26.** These are experimental `0.2.0-dev` models with numerical/runtime validation, not strength-qualified releases.

- Tested source: [`8aa329e240d2761f9a7ae9454a846be84388b016`](https://github.com/gaong5247-cmd/Stockless-variants/commit/8aa329e240d2761f9a7ae9454a846be84388b016).
- [Actual training run 40303979765](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979765): successful prepare job and five successful variant jobs; completed at 10:04:34 UTC.
- [Actual build/test run 40303979772](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979772): five successful platform jobs; completed at 10:03:30 UTC.
- [Machine-readable measurements and artifact identities](experiments/nnue-0.2.0-dev.json).
- [Independent downloaded-artifact integrity record](experiments/nnue-0.2.0-dev-integrity.json).
- [Architecture, training commands and model loading](NNUE.md).

The record identifies the actual code commit. Later documentation-only commits do not imply another training run.

## Download the independent models

| Variant | Positions generated | Games | Model file | Experiment artifact |
| --- | ---: | ---: | --- | --- |
| Chess | 2,199 | 32 | `chess.svnnue` | [Download](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979765/artifacts/12641956813) |
| Shogi | 2,255 | 32 | `shogi.svnnue` | [Download](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979765/artifacts/12640814306) |
| Janggi | 2,116 | 32 | `janggi.svnnue` | [Download](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979765/artifacts/12641588208) |
| Xiangqi | 2,142 | 32 | `xiangqi.svnnue` | [Download](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979765/artifacts/12640984271) |
| Makruk | 2,304 | 32 | `makruk.svnnue` | [Download](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979765/artifacts/12641523242) |

Each Actions archive contains `nnue-experiment/<variant>/` and the native NNUE regression report. The variant directory contains the `.svnnue` under `model/`, float checkpoint, metrics, heldout split, data/provenance manifests, positions and complete game histories, native runtime validation, training scripts and a matching Linux engine package with complete corresponding C++ source. Models are not automatically enabled.

The training artifacts have a recorded expiry of **2026-11-25** and may require GitHub sign-in. The measurements and SHA records above remain in this repository. For Windows or Android, use the matching `0.2.0-dev` build from [DOWNLOADS.md](DOWNLOADS.md); old `0.1.0-dev` binaries do not implement this format.

## What actually ran

Each target generated 32 independent games with a maximum of 80 plies, eight random legal opening plies, classical teacher depth 3 and seed `20261026`. Teacher settings were one thread, Hash 16 MiB, MultiPV 1, `StocklessHybrid=true`, `EvalBackend=Classical`, `Use NNUE=false` and Chess960 disabled. All teacher manifests identify a clean checkout of the tested source commit.

The CPU NumPy trainer ran **40 epochs per target**, batch size 128, a 64-unit hidden layer and independently initialized weights. It used Python 3.12.15, NumPy 2.3.5 and one BLAS/OMP/MKL thread on Linux x86-64 GitHub runners. CPU model/throughput was not measured; these results are not a hardware performance comparison.

Across all five games, generation produced **11,016 positions from 160 games**. There were **22 games ended by native rules and 138 truncated games**. Truncated outcomes are null, including all 32 Makruk games; they are never counted as draws. Current training uses teacher scores, not those game outcomes.

| Variant | Retained train / validation / test | Completed / truncated games | Actual gradient steps | Selected epoch |
| --- | --- | --- | ---: | ---: |
| Chess | 1,581 / 360 / 257 | 4 / 28 | 520 | 2 |
| Shogi | 1,570 / 315 / 360 | 3 / 29 | 520 | 36 |
| Janggi | 1,415 / 349 / 339 | 8 / 24 | 480 | 12 |
| Xiangqi | 1,467 / 310 / 359 | 7 / 25 | 480 | 1 |
| Makruk | 1,566 / 356 / 360 | 0 / 32 | 520 | 10 |

The retained counts follow whole-game splitting and exact feature-plus-side deduplication. There were **2,520 optimizer updates** in total. Network manifests and metrics record changed weight hashes, checkpoint choice, exact data/split hashes and model hashes. This is actual training and native inference, not a randomly generated file or a renamed upstream network.

Two Janggi games ended before producing a sampled row. They count toward native game outcomes, while only nonempty games enter the training partition.

## Approximation results and limits

Errors below are against heldout teacher labels clipped to ±2,000 **engine cp**. They are not Elo, general-chess pawn units for every variant, or a basis for comparing strength across games. The constant baseline predicts the mean training label on every test input.

| Variant | Test float MAE | Constant baseline MAE | Validation MSE improved from initialization? |
| --- | ---: | ---: | --- |
| Chess | 454.463 | 475.423 | Yes |
| Shogi | 535.774 | 586.528 | Yes |
| Janggi | 444.733 | 480.929 | Yes |
| Xiangqi | 346.681 | 354.360 | **No** |
| Makruk | 319.040 | 317.982 | Yes |

Xiangqi's selected trained checkpoint did not improve validation MSE over initialization. Makruk's test MAE is slightly worse than the constant baseline. The other four models have lower test MAE than that baseline on this one heldout split. Chess nevertheless has slightly higher test MSE/RMSE than its baseline. None of these measurements establishes a playing-strength gain. These shallow-teacher corpora are small; they do not justify automatically replacing the classical evaluator.

The feature limitations remain material: board/pocket features omit promoted origin, castling/en-passant rights, counters and history. Native rules still control legal moves and game results. Larger and more diverse data, better features/architectures and controlled matches are pending.

## Quantization and actual native inference

| Variant | Test float-to-integer MAE | Maximum error across validation and test | Native trained-model checks |
| --- | ---: | ---: | --- |
| Chess | 0.901 cp | 2.958 cp | 68 passed / 0 failed |
| Shogi | 1.066 cp | 5.264 cp | 68 passed / 0 failed |
| Janggi | 1.045 cp | 4.134 cp | 68 passed / 0 failed |
| Xiangqi | 0.477 cp | 1.913 cp | 68 passed / 0 failed |
| Makruk | 0.540 cp | 2.518 cp | 68 passed / 0 failed |

All models passed the predeclared validation/test gates: quantization MAE ≤3 cp, P99 ≤20 cp and maximum ≤40 cp. Each model then passed 64 heldout exact integer/feature/accumulator comparisons and four legal depth-3 searches with `StocklessNNUE` actually active: **340 native checks, zero failures**. The test split is checksum-bound and independently reconstructed from its seed before selecting positions.

Each training job also passed the 59-case neural regression suite and 20 data/split integrity tests. The native C++ accumulator harness checks make/unmake and lazy refresh separately; a numerically valid file alone is not sufficient to pass these gates.

## Platform build evidence

| Target | Build conclusion | Engine regression | Protocol validation | Neural regression |
| --- | --- | ---: | ---: | ---: |
| Linux x64 | Success | 214 passed | 24 passed | 59 passed |
| Linux x64 AVX2 | Success | 214 passed | 24 passed | 59 passed |
| Windows x64 | Success | 214 passed | 24 passed | 58 passed |
| Windows x64 AVX2 | Success | 214 passed | 24 passed | 58 passed |
| Android ARM64 API 26 | Success | Cross-build/ELF only | Not run | Compiled, not run |

Linux generic and AVX2 each also passed the C++ rule suite (149 cases, 43,037 checks, 4,692 make/unmake transitions), 2,581,152 variant-policy checks and the original Stockless policy regression. Windows omits the extra C++ neural harness, explaining its one fewer neural check. Its integer/file/search tests execute in the actual Windows binary. AVX2 is exercised, not merely compiled.

Android uses pinned NDK `27.2.12479018`, compiles the new neural runtime and passes the AArch64 ELF check. Physical-device inference, GUI integration and sustained play remain untested. The hosted training models were executed on Linux; platform regression fixtures and matching rule fingerprints are separate compatibility evidence.

## Artifact identities

All model descriptors use rules/feature fingerprint:

```text
055c294375a75423b80622757ed77a29175c76f33f66107ac223fb8da056e736
```

| Model | Bytes | SHA-256 |
| --- | ---: | --- |
| Chess | 100,520 | `9919dcdfa72423d8a735326f96f9f1ef205b2bce6cfafec9c3516cde69c03310` |
| Shogi | 210,600 | `e475d308eb9f3f0c401ce0bea55e6093fb0e61032eef8673c34efb287233b294` |
| Janggi | 163,752 | `1dcf7c5f723d4b9557379498fa286206af68ed4c380ce74644c399a9bff49bab` |
| Xiangqi | 163,752 | `16b4b1d49977d70c6baf720a34bc8a813191aa408916e449f7e0eb729f49511e` |
| Makruk | 100,520 | `ec26d93a485b923a7ec12a307af76c429695e8049c2c6f9ab3b4745a5ed85648` |

The machine-readable record includes SHA-256 values for **all five** models, every downloaded Actions archive, data/metrics/split/checkpoint manifests and the teacher. No dataset rows or binary weights are committed to Git. Model status remains `experimental`, `default_enabled=false`, `strength_validated=false`, with no Elo value.

After download, all ten Actions archives were independently inspected without executing their binaries. Every outer/inner package checksum, executable/model/data binding and all 81 corresponding-source file hashes per engine package matched. Reconstructing the rules fingerprint from the packaged C++/header contents, with line-ending normalization and the recorded notice prefix removed, produced the same fingerprint for every training and platform package. The integrity record reports inspection conditions separately from native engine test cases; it is not additional evidence of Android runtime execution.
