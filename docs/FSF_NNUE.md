# Official Fairy-Stockfish NNUE files in release ZIPs

The **FSF .nnue loader** is reused from the pinned Fairy-Stockfish rules/evaluation
engine, without converting network formats or changing feature encoding.
It is **not** the experimental Stockless SVNNUE-v1 network format.

## Included unmodified networks

The distribution packaging pipeline downloads these files **directly from the
links on [the Fairy-Stockfish NNUE page](https://fairy-stockfish.github.io/nnue/)**:

| Variant | Network filename |
| --- | --- |
| Chess | `nn-46832cfbead3.nnue` |
| Janggi (including compatible modes) | `janggi-9991472750de.nnue` |
| Shogi | `shogi-878ca61334a7.nnue` |
| Xiangqi | `xiangqi-c07e94a5c7cb.nnue` |
| Makruk | `makruk-a8c621e24a8c.nnue` |
| Crazyhouse | `crazyhouse-8ebf84784ad2.nnue` |

The exact original download URLs, network authors, and filenames are pinned in
`config/fsf-networks.json`. The download helper requires a SHA-256 digest
whose first 12 hex characters match the file's official hash suffix. It creates
`fsf-networks-manifest.json` with the **complete digest and file size**.
Missing, corrupt or replaced downloads fail the build rather than shipping a
placeholder or silently falling back.

This bundle is deliberately finite, not a claim of all the many FSF variants'
networks being included. Unsupported/other variants use classical evaluation.

## Folder layout after extracting the distribution ZIP

```text
stockless-variants.exe            # stockless-variants on Linux/Android
networks/
  nn-46832cfbead3.nnue
  janggi-9991472750de.nnue
  shogi-878ca61334a7.nnue
  xiangqi-c07e94a5c7cb.nnue
  makruk-a8c621e24a8c.nnue
  crazyhouse-8ebf84784ad2.nnue
fsf-networks-manifest.json
NETWORKS.md
manifest.json
...
```

When the engine starts beside `networks/` and the bundled **chess** NNUE is
present, its `Use NNUE` default is enabled; `EvalFile` lists all six
`networks/<filename>` entries with FSF's platform-specific separator.
Changing `UCI_Variant` automatically selects its matching network.

On **Windows** `EvalFile` separates file paths with `;` and on **Linux /
Android** with `:`. Don't add spaces around separators or rename networks.
For example, with an extracted bundle:

```text
uci
setoption name UCI_Variant value janggi
isready
position startpos
eval
go depth 8
```

The official FSF network-selection mechanism recognizes Janggi mode aliases
such as `janggimodern` from the variant's `nnueAlias`.

You can select another original FSF file with:

```text
setoption name UCI_Variant value shogi
setoption name EvalFile value shogi-customhash123.nnue
setoption name Use NNUE value true
```

The custom filename must actually exist and retain the variant prefix; replace
the illustrative filename with a real FSF network. Missing files are rejected
when enabling the network, not described as successfully loaded.

A **source build without the bundled networks** starts with `Use NNUE=false`
and keeps classical evaluation functional.

### Independent experimental SVNNUE-v1

```text
setoption name UCI_Variant value shogi
setoption name EvalFile value /path/to/valid-shogi.svnnue
setoption name EvalBackend value StocklessNNUE
```

Do **not** rename `.nnue` to `.svnnue` or the reverse: the binary architectures
are different. The `EvalBackend` option controls the experimental Stockless
backend; `Use NNUE` controls the original FSF backend.

## Building a network-inclusive archive

```bash
python -m pip install gdown==5.2.0
python tools/fsf_networks.py download
make build ARCH=x86-64 JOBS=2
python tools/fsf_networks.py verify
python tests/fsf_nnue_runtime_test.py --engine build/engine/src/stockless-variants --report build/fsf-nnue-tests.json
python tools/catalog_variants.py --engine build/engine/src/stockless-variants
python tools/package.py --engine build/engine/src/stockless-variants --platform linux --arch x86-64 --report build/fsf-nnue-tests.json
```

CI downloads networks once and shares the exact verified bytes with all five
platform builds. Windows and Linux are expected to execute each bundled network
through the real C++ FSF evaluation path. Android is cross-built; network
hash verification does not imply an on-device NNUE execution test.

## Attribution and permissions

Model authors and official links are recorded in `config/fsf-networks.json`
and `fsf-networks-manifest.json` in distributions. The FSF website explicitly
states CC0 terms for networks dated **2026 or later**; older model licenses
are not specified there. Please confirm redistribution permissions for older
networks before publicly publishing bundles. Engine source-code licensing is
documented separately in `NOTICE.md` and `LICENSES/`.
