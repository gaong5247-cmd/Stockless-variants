#!/usr/bin/env python3
"""Add native, synchronized training export to the generated engine only.

Copyright (C) 2026 Stockless Variants contributors
SPDX-License-Identifier: AGPL-3.0-only
"""
from pathlib import Path


def replace(path: Path, old: str, new: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"{path}: training overlay anchor changed: {old!r}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def apply(out: Path) -> None:
    uci = Path(out) / "src/uci.cpp"
    replace(uci, '#include "uci.h"\n',
            '#include "uci.h"\n#include "stockless_core/training_export.h"\n')
    anchor = '      else if (token == "compiler") sync_cout << compiler_info() << sync_endl;\n'
    replace(uci, anchor, '''      else if (token == "nnuefeatures" || token == "nnuefeaturesv2" || token == "nnuefeaturesv3")
      {
          // This command owns a stable root snapshot, including repetition history.
          // Joining also makes accidental requests during search safe.
          Threads.stop = true;
          Threads.main()->wait_for_search_finished();
          sync_cout << "info string NNUE_FEATURES "
                    << (token == "nnuefeaturesv3"
                        ? Stockless::training_features_v3_json(pos)
                        : Stockless::training_features_json(pos, token == "nnuefeaturesv2"))
                    << sync_endl;
      }
''' + anchor)
