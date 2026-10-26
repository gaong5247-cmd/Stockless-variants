# Development build downloads

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
