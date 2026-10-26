# Third-party notices and source provenance

Stockless Variants is an independent integration of the pinned Fairy-Stockfish
variant runtime with the actual Stockless v0.9.0 search-policy code. It is not
an official release of any upstream project.

| Component | Exact source revision | License | Role |
| --- | --- | --- | --- |
| [Fairy-Stockfish](https://github.com/fairy-stockfish/Fairy-Stockfish) | `9f778da667f6e07dae1e85d3e2ea204fc6dee94d` | GPL-3.0-or-later; preserve upstream headers and AUTHORS | C++ variant rules, position, move generation, search/runtime, classical evaluation, protocol implementation |
| [Stockless](https://github.com/gaong5247-cmd/Stockless) | `3bb0745daf161234143b04a3364d55b8b36fd26c` | AGPL-3.0 | Prepared hybrid search policy and settings; threat-signal contract |
| [Stockfish](https://github.com/official-stockfish/Stockfish) | Stockless reference pin `0a215d6c9e48856ef630013b8ab8312941a59057`; FSF also retains its historical Stockfish ancestry | GPL-3.0-or-later | Original engine ancestry and audit reference; the current Stockfish runtime/NNUE is not linked into this build |
| [Reckless](https://github.com/codedeliveryservice/Reckless) | Stockless reference pin `7300f044812d80397960e3a27a4f085e9487419a` | AGPL-3.0 | Search/threat research provenance inherited from Stockless; Rust engine and network are not linked or bundled |

New Stockless Variants integration code and Stockless-derived policy code use
AGPL-3.0. The original GPL-covered portions retain their GPL terms and copyright
notices. Section 13 of GPLv3 and AGPLv3 expressly permits combining these works;
AGPL network source-offer requirements apply as specified by that section.
This notice does not relicense the original GPL components.

The complete license texts are in `LICENSE`, `LICENSES/AGPL-3.0.txt`, and
`LICENSES/GPL-3.0.txt`. The materialized source retains upstream `Copying.txt`
and `AUTHORS`. Distributed binaries must be accompanied by the corresponding
source or an appropriate source offer/pointer that reproduces that exact build,
including the pinned sources, overlays, configuration, and build scripts.

## Modifications in Stockless Variants

- Port Stockless prepared search-policy decisions into the FSF search, adapting
  the fractional reduction scale and neutralizing modern-Stockfish inputs that
  are absent from FSF.
- Replace the orthodox 8x8 threat extractor with variant-aware board queries and
  objective-specific guards.
- Add hybrid-policy options, explicit evaluation-backend restrictions, strict
  transactional variant configuration checks, and protocol mutation guards.
- Validate position input and complete move lists before committing updates.
- Repair inherited gating/free-drop hash inconsistencies, protect signed virtual
  reserve counts in pocket hashing, and isolate caches when rule definitions
  change between searches.
- Correct free-drop pocket hash accesses, Seirawan pocket/material hashes,
  and omitted gating-right state in transposition/repetition keys. See
  `docs/RULE_FIXES.md`; modified source files carry dated change notices.
- Add capability metadata, rule/invariant/protocol tests, reproducible build
  tooling, and platform packaging.

No upstream logo, Chess.com artwork, private server code, or network weights are
included. No new Elo or strength claim is implied by reusing upstream code.

## Credits

Fairy-Stockfish: Fabian Fichter and contributors, including the contributors
listed in its preserved AUTHORS file. Stockfish: the Stockfish developers and
Glaurung ancestry, as recorded by upstream. Stockless: @gaong5247-cmd,
acknowledged contributor @eueong04-create, and other project contributors.
Reckless: its upstream authors and contributors; original notices remain
authoritative.
