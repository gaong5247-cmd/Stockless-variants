# Development build downloads

## Current 0.2.0-dev builds and independent NNUE models

The [five-target build run](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979772)
for commit `8aa329e240d2761f9a7ae9454a846be84388b016` completed successfully on
2026-10-26. These remain experimental development packages, with classical
evaluation enabled by default.

| Target | Download | Actual validation |
| --- | --- | --- |
| Windows x64 | [Generic CPU artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979772/artifacts/12641907203) | 214 engine, 24 protocol, 58 neural checks passed |
| Windows x64 AVX2 | [AVX2 artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979772/artifacts/12640784007) | Same counts; requires AVX2 |
| Linux x64 | [Generic CPU artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979772/artifacts/12641124362) | C++ rules/policies plus 214 engine, 24 protocol, 59 neural checks passed |
| Linux x64 AVX2 | [AVX2 artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979772/artifacts/12641737790) | Same counts with actual AVX2 execution; requires AVX2 |
| Android ARM64 | [ARM64 API 26 artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40303979772/artifacts/12641682954) | NDK build, AArch64 ELF and package checks; device execution pending |

The separately [trained Chess/Shogi/Janggi/Xiangqi/Makruk models](NNUE_RESULTS.md)
are available in their experiment artifacts, together with training data,
metrics, verification and matching Linux engine/source packages. No neural
file is embedded in the platform engine packages. See [NNUE.md](NNUE.md) for
loading a compatible `.svnnue` explicitly; old `0.1.0-dev` executables cannot
load it.

Unpack the Actions ZIP, then the package under `dist/`. The package includes
the exact executable, corresponding C++ source, notices, supported-rule catalog,
checksums and platform-specific validation reports. Linux/Android may need
`chmod +x stockless-variants` after extraction. Android is a native executable,
not an APK. GitHub sign-in may be required for artifact downloads.

## Historical 0.1.0-dev baseline

**Version 0.1.0-dev — experimental engineering preview.** These artifacts are
from the successful [five-target CI run](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40297053245)
for commit `1ace3f06ffa878a4e4fd0e35985f07095224255d`.
This is not a stable release or a claim of complete support for the universal
engine specification.

| Target | Download | Validation boundary |
| --- | --- | --- |
| Windows x64 | [Generic CPU artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40297053245/artifacts/12639240489) | Runtime rules/protocol tests passed |
| Windows x64 AVX2 | [AVX2 artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40297053245/artifacts/12638238459) | Runtime tests passed; requires an AVX2-capable CPU |
| Linux x64 | [Generic CPU artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40297053245/artifacts/12638966567) | C++ and runtime tests passed |
| Linux x64 AVX2 | [AVX2 artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40297053245/artifacts/12639065945) | C++ and runtime tests passed; requires an AVX2-capable CPU |
| Android ARM64 | [ARM64 API 26 artifact](https://github.com/gaong5247-cmd/Stockless-variants/actions/runs/40297053245/artifacts/12638213195) | Cross-build and ELF checks passed; device/GUI execution pending |

GitHub may require sign-in to download Actions artifacts. Their recorded expiry
is **2027-01-24**; the pinned source and build workflow remain in this repository.
No stable GitHub Release has been published.

## Contents

Unpack the downloaded Actions artifact, then unpack the engine package ZIP in
its `dist/` directory. Each engine package includes:

- The native executable (`stockless-variants.exe` on Windows).
- Version, source commit, architecture, executable/file SHA-256 hashes and
  explicit development/runtime-validation status in `manifest.json`.
- The exact generated C++ source in `corresponding-source.zip`, with a build
  recipe and original copyright/license notices.
- Source provenance, licenses, custom examples, the variant inventory and docs.
- Passing reports for the exact tested executable on Linux/Windows. Android
  deliberately has no runtime report.

No NNUE network is bundled. Evaluation is classical. Android's artifact is an
engine executable for a compatible host, not an APK; see [ANDROID.md](ANDROID.md).
After extracting on Linux/Android, executable permissions may need restoring
with `chmod +x stockless-variants`.

The [validation record](VALIDATION.md) describes actual coverage. Read
[LIMITATIONS.md](LIMITATIONS.md) for unfinished rules, search architectures and
unsupported games before selecting a variant.
