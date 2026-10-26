# Recorded validation of the first executable milestone

Recorded on **2026-10-26**. This is bounded engineering evidence for
`0.1.0-dev`, not a stable-release approval, an Elo result or certification of
every regional rule convention.

## Exact build

- Engine source commit: `f985a5b187defbbac52c581a8518d96d3df69f46`.
- The tree was clean when sources were materialized. Later documentation,
  process-isolation test and platform-neutral package-path changes do not alter
  this tested Linux C++ source.
- Fairy-Stockfish pin: `9f778da667f6e07dae1e85d3e2ea204fc6dee94d`.
- Stockless pin: `3bb0745daf161234143b04a3364d55b8b36fd26c`.
- Linux x86-64, GCC 13.3.0, C++17, `largeboards=yes all=yes nnue=no`.
- Default classical evaluation; one search thread unless a case explicitly tests two.
- Engine SHA-256: `7dbb3e00afbfc04ef80875852d23ca14b75341845b8cad31666de49c24bb3bef`.

The host was a Linux virtual environment exposing nine AMD EPYC 9V74 CPUs.
Timing from this environment must not be represented as a hardware-independent
performance result. Machine details and sanitizer build identity are in the
[core summary](validation/linux-core-summary.json).

The commit above is the local core-build identity. Publication through the
connected GitHub app produced public commit
[`1ace3f06ffa878a4e4fd0e35985f07095224255d`](https://github.com/gaong5247-cmd/Stockless-variants/commit/1ace3f06ffa878a4e4fd0e35985f07095224255d).
Its complete tree, `cb4a7666349bce8880d03470e04e345fdc8bd22a`, exactly matches
the locally prepared publication tree. The public commit was independently
rebuilt and tested on hosted runners as recorded below.

## Completed checks

| Suite | Result | Scope |
| --- | ---: | --- |
| Black-box engine suite | **355 passed, 0 failed** | 6 protocol cases, 155 perft cases, 50 search cases, 8 bounded playouts, 136 differential cases |
| Strict protocol/configuration suite | **24 passed, 0 failed** | Bad options/FEN/SFEN/moves, invalid INI, transactional loading, variant-switch cache isolation |
| C++ rule/state suite | **149 cases; 43,037 checks; 4,692 state transitions; 0 failures** | Legal moves, special rules, outcomes, FEN rebuild, make/unmake and incremental hash/material/pawn state |
| Variant search-policy suite | **2,581,152 checks; 0 failures** | Native attacks, royal profiles, policy safety gates, deterministic depth quantization and bounds |
| Original Stockless policy regression | **All passed** | 200,000 prepared-move, 100,000 single-call, 4 mutable-depth and 100,201 mobile Overdrive cases |
| ASan + UBSan C++ suites | **Same rule/policy counts passed** | Instrumented rule/state and policy executables plus original Stockless policy test |
| ASan + UBSan actual searches | **11 passed** | Depth 3 in Bughouse, Supply, Ataxx, Seirawan, Shouse, Atomic, Crazyhouse, Shogi, Janggi, Xiangqi and Duck |
| Default engine `bench` | **47 positions completed** | 7,003,664 nodes; completion smoke, not a playing-strength measurement |
| Runtime inventory | **121 IDs** | All compiled registrations advertised; IDs include aliases |
| INI validation | **Passed** | The full 143-section upstream examples file and the project's custom examples |

The source catalog contains **121 compiled registration IDs** and **143 INI
examples**. The 63 requested/additional game entries classify as 44 Native,
7 Adapted, 2 New Rules and 10 Special Architecture. These are source/architecture
categories; they are not 63 independently certified games. No catalog entry is
marked release-approved.

### Comparison with the unmodified reference

A separately compiled, unmodified pinned Fairy-Stockfish executable was used
for legal-move-set and bounded-line comparisons. Its SHA-256 is
`d3f863c430d739d159d0dc74e818d7df914900b518594d035eb7a4cbef653753`.

With `StocklessHybrid=false`, seven fresh-process baseline searches match the
original reference's **best move, score and searched-node count exactly**. The
fixtures cover chess, Atomic, Crazyhouse, Shogi, Janggi, Xiangqi and Makruk.
Each fixture starts fresh processes because upstream `ucinewgame` does not clear
all variant-dependent pawn/material caches. The new engine additionally fixes
that variant-switch cache problem; the dedicated regression checks warm-switch
behavior against a fresh process.

With the hybrid policy enabled, targeted desktop/mobile checks verify a real
search effect. This establishes that policy options reach the search, not that
the effect improves playing strength. Overdrive also has explicit boundedness
and mobile-profile checks.

### Sanitizer boundary

AddressSanitizer and UndefinedBehaviorSanitizer ran with errors configured to
halt. LeakSanitizer could not read process-task information in this sandbox, so
the completed runs used `ASAN_OPTIONS=detect_leaks=0:halt_on_error=1`. **Memory
leak detection was not completed.** No leak-free or race-free claim is made.

The instrumented searches specifically exercise the inherited negative pocket
counts that previously triggered an out-of-bounds key lookup in Bughouse and
Supply. See [correctness repairs](RULE_FIXES.md) for the actual patches.

## Reports and reproduction

- [Complete black-box report](validation/linux-engine-tests.json)
- [Complete input/configuration report](validation/linux-protocol-tests.json)
- [Build, C++ and sanitizer summary](validation/linux-core-summary.json)
- [C++ test log](validation/cpp-tests.log)
- [Instrumented C++ test log](validation/sanitized-cpp-tests.log)
- [Instrumented search log](validation/sanitized-search.log)

```bash
make build ARCH=x86-64 JOBS=2
python3 tools/compile_rule_tests.py
python3 tests/run_engine_tests.py \
  --engine build/engine/src/stockless-variants \
  --reference build/reference/src/stockfish \
  --full-perft --timeout 120 --report build/engine-tests.json
python3 tests/protocol_validation_test.py \
  --engine build/engine/src/stockless-variants --report build/protocol-tests.json
```

The optional reference must be built separately from the pinned, unmodified
upstream; omit `--reference` when that executable is unavailable. To reproduce
instrumented C++ coverage:

```bash
python3 tools/materialize.py --output build/sanitized
make -C build/sanitized/src -j2 build ARCH=x86-64 \
  largeboards=yes all=yes nnue=no debug=yes optimize=no \
  sanitize=undefined,address EXE=stockless-variants
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
python3 tools/compile_rule_tests.py --source build/sanitized/src \
  --extra-flag=-fsanitize=address,undefined
```

## Platform and release boundaries

**All five jobs passed** in
[Actions run 40297053245](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40297053245),
building public commit `1ace3f06ffa878a4e4fd0e35985f07095224255d`.
The run completed on 2026-10-26 at 09:09:07 UTC.

| Hosted target | Actual result |
| --- | --- |
| Linux x64 | C++ suites above, 214 black-box cases, 24 input/configuration checks, package verification: passed |
| Linux x64 AVX2 | Same C++ and runtime suites, package verification: passed |
| Windows x64 | 214 black-box cases, 24 input/configuration checks, package verification: passed |
| Windows x64 AVX2 | Same runtime suites, package verification: passed |
| Android ARM64 API 26 | NDK 27.2.12479018 cross-compilation, AArch64 ELF and package verification: passed |

The CI black-box count is lower than the local 355-case run because CI does not
build the independent reference executable or enable the largest `--full-perft`
cases. Both scopes are preserved rather than conflated. Windows runs the actual
executable but not the Linux C++ harness. Android does **not** run on a physical
device or Android GUI in this workflow.

The Linux generic, Windows generic and Android artifacts were then downloaded
and their packaged manifest file hashes independently checked. Report hashes
match the Linux/Windows executable hashes. Windows is PE32+ x86-64; Android is
ELF64 AArch64 and explicitly records `runtime_tested_in_this_package=false`.
The corresponding-source archives preserve the platform build recipes.
See the [machine-readable CI record](validation/ci-run-40297053245.json) and
[development downloads](DOWNLOADS.md).

Packages bind reports/catalogs to the exact executable hash and supply complete
corresponding generated source. A failed or mismatched report cannot be used
as package validation evidence. Source/architecture limitations and unverified
tournament rules remain explicit in [LIMITATIONS.md](LIMITATIONS.md).
