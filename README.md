# Stockless Variants

**An experimental C++ variant engine combining Fairy-Stockfish's rule-aware runtime with the actual Stockless v0.9.0 hybrid search policy.**

This repository is independent of [Stockless](https://github.com/gaong5247-cmd/Stockless). It does not change that project's main branch or releases. Version `0.2.0-dev` adds an independent experimental SVNNUE-v1 runtime and CPU training tools to the first executable milestone. **It remains a development engine without strength certification or stable-release approval.**

**[Previous 0.1.0-dev build artifacts](docs/DOWNLOADS.md)** ·
[Recorded baseline validation](docs/VALIDATION.md) ·
[Experimental NNUE and training](docs/NNUE.md) ·
[Supported-rule inventory](docs/VARIANTS.md) ·
[Known limitations](docs/LIMITATIONS.md)

## What is implemented

- Pinned, reproducible source materialization from two direct Git submodules; no changes to their source trees.
- Large-board/all-variant build of the FSF rule core, including chess, Chess960, Atomic, Crazyhouse, Duck, Shogi, Janggi, Xiangqi, Makruk and custom Betza/INI variants.
- Byte-for-byte reuse of Stockless v0.9.0's configuration and prepared search-policy headers, with a new variant-aware threat extractor and a bounded fixed-point-to-integral-depth adapter.
- Real search changes: late-move-reduction buyback, conservative null-move gating, and opt-in bounded desktop overdrive on the orthodox profile. The mobile profile retains effective sub-ply policy signals through deterministic quantization.
- UCI/UCI_Variant, USI, UCCI and CECP/XBoard through the existing FSF adapters, with checked options and transactional position/configuration loading.
- Classical evaluation by default, with opt-in support for separately trained SVNNUE-v1 models for native `chess`, `shogi`, `janggi`, `xiangqi` and `makruk`. Binary headers bind variant, architecture, quantization and rule/feature-source identity.
- Native feature/self-play export, a CPU NumPy trainer, whole-game train/validation/test separation, quantization checks, incremental accumulators and native trained-model validation. Training errors measure teacher approximation, not playing strength.
- Automatic extraction of compiled registrations and INI examples, with requested-game classification and separate validation evidence.
- Rule, make/unmake/hash, perft, protocol, bounded-play and policy regression tools; Linux/Windows/Android build jobs and corresponding-source packages.

The [generated catalog](docs/VARIANTS.md) distinguishes **source implementation, runtime availability, testing and unresolved rule differences**. Registration IDs include aliases. A game appearing in that list is not a claim of tournament-rule certification.

## Build

Requirements: Python 3.10+, Git, GNU Make and a C++17 compiler. Building and running the default classical engine requires no model download or third-party Python package. The optional CPU trainer requires Python 3.11+ and the separately pinned NumPy dependency in `tools/nnue/requirements.txt`.

```bash
git clone https://github.com/gaong5247-cmd/Stockless-variants.git
cd Stockless-variants
git submodule update --init --depth 1
make build ARCH=x86-64 JOBS=2
```

Executable: `build/engine/src/stockless-variants`.

Use `ARCH=x86-64-avx2` only on a compatible CPU. To rebuild without rematerializing unchanged sources:

```bash
make -C build/engine/src -j2 build ARCH=x86-64 largeboards=yes all=yes nnue=no EXE=stockless-variants
```

The direct submodules are sufficient. Recursively downloading Stockless's Stockfish/Reckless research dependencies is unnecessary for this variant build. `tools/materialize.py` checks the exact pins and rejects modified upstream trees or changed patch anchors.

The Make option `nnue=no` disables upstream network download/embedding; the independent SVNNUE-v1 runtime is still compiled.

### Windows

Use MSYS2 UCRT64 with `make` and `mingw-w64-ucrt-x86_64-gcc`, plus Python:

```bash
python tools/materialize.py
make -C build/engine/src -j2 build COMP=mingw ARCH=x86-64 largeboards=yes all=yes nnue=no EXE=stockless-variants.exe
```

### Android ARM64

See [Android instructions](docs/ANDROID.md). The artifact is a native engine executable for a compatible host GUI, not an APK or standalone graphical application. CI cross-compilation and ELF inspection do not prove execution on a physical Android device.

## Run

```text
uci
setoption name UCI_Variant value janggi
setoption name StocklessHybrid value true
setoption name StocklessAggression value 145
isready
position startpos
go depth 6
```

For Japanese notation, start with `usi`; for Chinese-engine notation, use `ucci`. Details and examples are in [PROTOCOLS.md](docs/PROTOCOLS.md).

### Experimental variant NNUE

After producing a compatible model with the [training and validation tools](docs/NNUE.md), select its native variant, load its `.svnnue` file and explicitly activate the backend:

```text
uci
setoption name UCI_Variant value shogi
setoption name EvalFile value /path/to/shogi.svnnue
setoption name EvalBackend value StocklessNNUE
position startpos
nnueeval
go depth 4
```

The five targets use separate models. Stockfish/FSF `.nnue` files, Chess960 mode and custom rule clones are outside SVNNUE-v1 compatibility. Its compact board/pocket features omit promoted-piece origin, castling/en-passant rights and rule/history counters; native rules still govern legal moves and game results. `Classical` remains the default, and no strength improvement or Elo is claimed. The training workflow's definition and any successful run must be distinguished by their recorded artifacts and reports.

### Custom variants

```text
setoption name VariantConfigPath value config/examples.ini
setoption name VariantName value stockless-7check
position startpos
go depth 4
```

Offline validation returns a nonzero exit status on errors:

```bash
build/engine/src/stockless-variants check config/examples.ini
build/engine/src/stockless-variants check upstream/fairy-stockfish/src/variants.ini
```

Loading a bad configuration must not partially register its variants. FSF's supported INI inheritance and Betza syntax are reused; unsupported grammar and rule combinations are rejected rather than silently treated as implemented.

## Validate and package

```bash
python tools/compile_rule_tests.py
python tests/run_engine_tests.py --engine build/engine/src/stockless-variants --report build/engine-tests.json
python tests/protocol_validation_test.py --engine build/engine/src/stockless-variants
python tools/catalog_variants.py --engine build/engine/src/stockless-variants --validation build/engine-tests.json
python tools/package.py --engine build/engine/src/stockless-variants --platform linux --arch x86-64 --report build/engine-tests.json
```

Optional `--reference /path/to/unmodified-fsf` compares actual legal move sets with a separately built pinned baseline. `--full-perft` includes the largest upstream reference cases. Test reports and complete corresponding source are delivered alongside binary packages; tests are evidence for their covered conditions, not Elo measurements. The [recorded local validation](docs/VALIDATION.md) identifies the exact tested build and scope.

## Architecture and scope

| Layer | Actual implementation |
| --- | --- |
| Variant registry, geometry, pieces, legal moves, rule outcomes | Fairy-Stockfish `VariantMap`, `Variant`, `PieceMap`, `Position`, `movegen` |
| PVS/alpha-beta, iterative deepening, TT, threads, time management | Fairy-Stockfish's variant-aware runtime |
| Prepared tactical hybrid policy | Actual Stockless v0.9.0 `search_policy.h` and `config.h` |
| Variant attack signals and safety profiles | New Stockless Variants adapter |
| Evaluation | FSF classical/variant evaluator by default; independent optional SVNNUE-v1 runtime and per-variant CPU training |
| Four-player, hidden-information and complete multi-board coordination | Not implemented |

This deliberately preserves FSF's existing rule-sensitive search instead of pretending that modern Stockfish's entire 8×8 runtime can be copied over unchanged. Stockless's current Stockfish NNUE is incompatible with this core. The original Stockless time-management/TT implementations are **not** independently transplanted in this milestone.

See [upstream audit](docs/UPSTREAM_AUDIT.md), [search design](docs/SEARCH.md), [Chess.com mapping](docs/CHESSCOM_RULES.md), [limitations](docs/LIMITATIONS.md) and [development roadmap](docs/ROADMAP.md).

## Licenses

Stockless-derived/new integration is AGPLv3. Fairy-Stockfish/Stockfish files retain their GPLv3-or-later notices. The complete license texts, provenance and exact revisions are in [LICENSE](LICENSE), [LICENSES](LICENSES) and [NOTICE.md](NOTICE.md). Reckless is an upstream research reference; its Rust engine and neural network are not linked here.

No Chess.com graphics, trademarks or private server code are included.

## 한국어

이 저장소는 **FSF의 변형 규칙 코어에 실제 Stockless v0.9.0 탐색 정책을 연결한 첫 개발 버전**입니다. 장기·쇼기·샹치 등은 기존 규칙 구현을 활용하고, 기물 이동에 맞는 위협 신호와 탐색 깊이 조절을 추가합니다. 원본 Stockless 저장소와 정식 릴리스는 변경하지 않습니다.

게임 목록은 실행 가능 여부와 검증 범위를 분리합니다. `0.2.0-dev`에는 체스·쇼기·장기·샹치·마크룩을 각각 별도 모델로 학습하는 CPU NumPy 파이프라인과 독립적인 SVNNUE-v1 추론·양자화·검증 기능을 추가했습니다. 기본 평가는 수작업 평가 함수이며, 모델을 명시적으로 불러와야 신경망 평가가 활성화됩니다. 학습 오차를 기력이나 Elo로 표현하지 않습니다.

4인 체스, 완전한 2보드 Bughouse, Fog of War, Spell Chess 및 GUI는 아직 구현되지 않았습니다. 기존 `0.1.0-dev` 다운로드와 검증 기록은 당시 빌드의 이력이며 새 모델의 검증 결과를 대신하지 않습니다.
