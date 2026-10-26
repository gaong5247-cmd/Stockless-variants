#!/usr/bin/env python3
"""Inventory pinned FSF rules without treating an INI example as a built-in.

This is a source inventory, not a C++ interpreter or a rules certification.
Use --engine for the actual UCI_Variant list; test evidence is kept separate.
Python 3.9+, standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CPP_TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/', re.S)
MEMBER = r"[A-Za-z_]\w*(?:\[[^\]]+\])*"
ASSIGNMENT = re.compile(r"v->(" + MEMBER + r")\s*(\|=|&=|\+=|-=|=)\s*([^;]+);", re.S)
CHAIN = re.compile(r"\s*v->(" + MEMBER + r")\s*=\s*")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_cpp(text: str, strings: bool = False) -> str:
    def replace(match: re.Match) -> str:
        value = match.group(0)
        if strings or value.startswith("//") or value.startswith("/*"):
            return "".join("\n" if char == "\n" else " " for char in value)
        return value
    return CPP_TOKENS.sub(replace, text)


def closing_brace(masked: str, opening: int) -> int:
    depth = 1
    for index in range(opening + 1, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if not depth:
                return index
    raise ValueError("Unclosed C++ function body")


def source_functions(source: str) -> dict:
    masked = mask_cpp(source, strings=True)
    comments_removed = mask_cpp(source)
    functions = {}
    for match in re.finditer(r"\bVariant\s*\*\s*(\w+)\s*\(\s*\)\s*\{", masked):
        end = closing_brace(masked, match.end() - 1)
        body = comments_removed[match.end():end]
        parent = re.search(r"Variant\s*\*\s*v\s*=\s*(\w+)\s*\(\s*\)\s*->\s*init\s*\(", body)
        assignments = []
        for assignment in ASSIGNMENT.finditer(body):
            fields = [assignment.group(1)]
            expression = assignment.group(3)
            if assignment.group(2) == "=":
                while chain := CHAIN.match(expression):
                    fields.append(chain.group(1))
                    expression = expression[chain.end():]
            for field in fields:
                assignments.append({"field": field, "operator": assignment.group(2),
                                    "expression": " ".join(expression.split())})
        pieces = [" ".join(item.split()) for item in
                  re.findall(r"v->((?:add_piece|remove_piece|reset_pieces)\s*\([^;]*\));", body)]
        functions[match.group(1)] = {
            "parent_function": parent.group(1) if parent else None,
            "line": source.count("\n", 0, match.start()) + 1,
            "assignments": assignments,
            "piece_operations": pieces,
        }
    return functions


def inherited_fields(name: str, functions: dict, trail: tuple = ()) -> dict:
    if name in trail:
        raise ValueError("Cyclic C++ variant inheritance: " + " -> ".join(trail + (name,)))
    if name not in functions:
        raise ValueError("Missing variant factory: " + name)
    function = functions[name]
    parent = function["parent_function"]
    fields = inherited_fields(parent, functions, trail + (name,)) if parent else {}
    for assignment in function["assignments"]:
        field = assignment["field"]
        value = assignment["expression"]
        operator = assignment["operator"]
        if operator != "=":
            previous = fields.get(field, {}).get("expression", "<header default>")
            value = "(" + previous + ") " + operator[:-1] + " (" + value + ")"
        fields[field] = {"expression": value, "source_function": name}
    return fields


def geometry(fields: dict, defaults: dict) -> dict:
    result = dict(defaults)
    for name, key in (("maxRank", "ranks"), ("maxFile", "files")):
        if name not in fields:
            continue
        raw = fields[name]
        raw = raw["expression"] if isinstance(raw, dict) else raw
        value = raw.strip()
        if re.fullmatch(r"\d+", value):
            result[key] = int(value)
        elif name == "maxRank" and re.fullmatch(r"RANK_\d+", value):
            result[key] = int(value[5:])
        elif name == "maxFile" and re.fullmatch(r"(?:FILE_)?[a-lA-L]", value):
            result[key] = ord(value[-1].lower()) - ord("a") + 1
        else:
            result[key] = None
    return result


def registrations(source: str, functions: dict, defaults: dict) -> list:
    comments_removed = mask_cpp(source)
    start = re.search(r"void\s+VariantMap::init\s*\(\s*\)\s*\{", comments_removed)
    if not start:
        raise ValueError("VariantMap::init() not found")
    end = closing_brace(mask_cpp(source, strings=True), start.end() - 1)
    body = comments_removed[start.end():end]
    offset = source.count("\n", 0, start.end()) + 1
    guards = []
    result = []
    for number, line in enumerate(body.splitlines(), offset):
        directive = re.match(r"\s*#\s*(ifdef|ifndef|if|elif|else|endif)\b\s*(.*)", line)
        if directive:
            kind, condition = directive.groups()
            if kind in ("ifdef", "ifndef", "if"):
                guards.append(("!" if kind == "ifndef" else "") + condition.strip())
            elif kind == "else":
                guards[-1] = "!(" + guards[-1] + ")"
            elif kind == "elif":
                guards[-1] = "elif " + condition.strip()
            elif kind == "endif":
                guards.pop()
            continue
        match = re.search(r'\badd\s*\(\s*"([^"\n]+)"\s*,\s*(\w+)\s*\(\s*\)\s*\)', line)
        if not match:
            continue
        variant_id, function = match.groups()
        fields = inherited_fields(function, functions)
        two_boards = fields.get("twoBoards", {}).get("expression") == "true"
        result.append({
            "id": variant_id,
            "origin": "compiled_registration",
            "factory": function,
            "source_line": number,
            "build_guards": list(guards),
            "factory_parent": functions[function]["parent_function"],
            "geometry_hint": geometry(fields, defaults),
            "rule_expressions": fields,
            "piece_operations": functions[function]["piece_operations"],
            "needs_external_board_coordinator": two_boards,
            "validation_evidence": [],
            "release_approved": False,
        })
    if not result or len({item["id"] for item in result}) != len(result):
        raise ValueError("No registrations or duplicate compiled variant IDs")
    return result


def ini_examples(text: str, builtins: list, defaults: dict) -> list:
    sections = []
    current = None
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        match = re.fullmatch(r"\[([^:\]]+)(?::([^\]]+))?\]", line)
        if match:
            name, parent = match.groups()
            current = {"id": name.strip(), "origin": "ini_example", "source_line": number,
                       "parent": parent.strip() if parent else None, "properties": {},
                       "validation_evidence": [], "release_approved": False}
            sections.append(current)
        elif "=" in line and current is not None:
            key, value = line.split("=", 1)
            key = key.strip()
            if key in current["properties"]:
                current.setdefault("duplicate_properties", []).append({
                    "key": key, "line": number, "previous": current["properties"][key],
                    "replacement": value.strip(),
                    "semantics": "last assignment wins, matching upstream Config map",
                })
            current["properties"][key] = value.strip()
        else:
            raise ValueError(f"Unrecognized active INI syntax at line {number}: {line}")
    by_id = {item["id"]: item for item in sections}
    if len(by_id) != len(sections):
        raise ValueError("Duplicate INI section")
    builtin_map = {item["id"]: item for item in builtins}

    def resolve(item: dict, trail: tuple = ()) -> tuple:
        name = item["id"]
        if name in trail:
            raise ValueError("Cyclic INI inheritance: " + " -> ".join(trail + (name,)))
        parent = item["parent"]
        fields = {}
        board = dict(defaults)
        chain = []
        if parent in by_id:
            fields, board, chain = resolve(by_id[parent], trail + (name,))
            item["parent_origin"] = "ini_example"
            if by_id[parent]["source_line"] >= item["source_line"]:
                item["load_warning"] = "Parent must be loaded before child by FSF"
        elif parent in builtin_map:
            fields = dict(builtin_map[parent]["rule_expressions"])
            board = dict(builtin_map[parent]["geometry_hint"])
            chain = [parent]
            item["parent_origin"] = "compiled_registration"
        elif parent:
            item["parent_origin"] = "unresolved"
            item["load_warning"] = "Unresolved parent: " + parent
        else:
            item["parent_origin"] = "default_template"
        fields.update(item["properties"])
        board = geometry(item["properties"], board)
        return fields, board, chain + [name]

    for item in sections:
        fields, board, chain = resolve(item)
        item["inheritance_chain"] = chain
        item["geometry_hint"] = board
        item["effective_property_keys"] = sorted(fields)
        item["requires_explicit_loading"] = True
        item["rule_validation"] = "not_performed_by_catalog"
    return sections


def runtime_inventory(engine: Path, config_paths: list, timeout: float) -> dict:
    commands = ["setoption name VariantPath value " + str(path.resolve()) for path in config_paths]
    commands.extend(["uci", "quit"])
    completed = subprocess.run([str(engine.resolve()), "noautoload"],
                               input="\n".join(commands) + "\n", cwd=engine.resolve().parent,
                               text=True, capture_output=True, timeout=timeout, check=False)
    if completed.returncode:
        raise ValueError(f"Engine exited {completed.returncode}: {completed.stderr[-2000:]}")
    option = next((line for line in completed.stdout.splitlines()
                   if line.startswith("option name UCI_Variant type combo ")), None)
    if option is None or "uciok" not in completed.stdout.splitlines():
        raise ValueError("Engine did not complete UCI handshake with UCI_Variant")
    diagnostics = [line for line in (completed.stdout + completed.stderr).splitlines()
                   if re.search(r"\b(error|invalid|failed|unsupported|unknown)\b", line, re.I)]
    if config_paths and diagnostics:
        raise ValueError("Config loading reported diagnostics: " + " | ".join(diagnostics[:10]))
    ids = re.findall(r"\bvar\s+(\S+)", option)
    if not ids:
        raise ValueError("Empty runtime variant list")
    return {"engine_sha256": digest(engine),
            "identity": [line[8:] for line in completed.stdout.splitlines() if line.startswith("id name ")],
            "loaded_configurations": [{"name": path.name, "sha256": digest(path)} for path in config_paths],
            "variant_ids": sorted(set(ids)),
            "evidence_kind": "protocol_inventory_only_not_rule_validation"}


def validation_reports(paths: list, runtime: dict | None) -> tuple:
    """Read the run_engine_tests.py schema; retain failures and exact core IDs."""
    reports = []
    by_variant = {}
    basenames = set()
    for path in paths:
        if path.name in basenames:
            raise ValueError("Validation reports need distinct filenames: " + path.name)
        basenames.add(path.name)
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("Validation report is not an object: " + path.name)
        tested_hash = report.get("engine_sha256")
        matches = tested_hash == runtime["engine_sha256"] if runtime and tested_hash else None
        if matches is False:
            raise ValueError("Validation report was produced by another executable: " + path.name)
        reports.append({"name": path.name, "sha256": digest(path), "report": report,
                        "matches_runtime_engine": matches})
        for test in report.get("tests", []):
            if not isinstance(test, dict):
                continue
            name = test.get("name", "")
            details = test.get("evidence", {})
            if not isinstance(details, dict):
                details = {}
            variant = details.get("variant") or details.get("default_variant")
            # Failed tests have no evidence dict; their names preserve the variant.
            if not variant and isinstance(name, str):
                prefixes = ("perft/", "playout/", "differential/startpos/", "differential/playout/")
                for prefix in prefixes:
                    if name.startswith(prefix):
                        variant = name[len(prefix):].split("/", 1)[0]
                        break
                if not variant:
                    variant = {"protocol/uci": "chess", "protocol/usi": "shogi",
                               "protocol/ucci": "xiangqi", "protocol/xboard": "chess"}.get(name)
                if not variant and name.startswith("search/") and name.count("/") == 1:
                    candidate = name.split("/", 1)[1]
                    if candidate not in ("two-threads", "hybrid-has-real-effect"):
                        variant = candidate
            if not isinstance(variant, str):
                continue
            entry = {"report": path.name, "ruleset_tested": variant,
                     "engine_sha256": tested_hash, "matches_runtime_engine": matches,
                     "test": name, "passed": test.get("passed") is True,
                     "evidence": details}
            if "error" in test:
                entry["error"] = test["error"]
            by_variant.setdefault(variant, []).append(entry)
    return reports, by_variant


def source_revision(source_root: Path) -> str | None:
    location = subprocess.run(["git", "-C", str(source_root), "rev-parse", "--show-toplevel"],
                              capture_output=True, text=True, check=False)
    if location.returncode or Path(location.stdout.strip()).resolve() != source_root.resolve():
        return None  # A copied source directory must not inherit its parent's repository SHA.
    completed = subprocess.run(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                               capture_output=True, text=True, check=False)
    return completed.stdout.strip() if not completed.returncode else None


def markdown(catalog: dict) -> str:
    def cell(value) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["# Variant inventory", "",
             "Generated by `tools/catalog_variants.py` from the pinned upstream files.", "",
             "**Inventory is not certification.** `Native` means a source rule implementation exists. "
             "Runtime advertisement, rule tests, and release approval are separate evidence. "
             "No game is promoted to release-approved by this script.", "",
             "## Source and architecture", "",
             f"- Fairy-Stockfish commit: `{catalog['upstream_commit']}`",
             f"- Compiled registrations in source: {len(catalog['compiled_registrations'])} (includes aliases).",
             f"- INI example sections: {len(catalog['ini_examples'])}; these require explicit loading.",
             "- Core limit: two players, rectangular boards up to 12 files × 10 ranks with LARGEBOARDS.",
             "- Source geometry and rule expressions are extracted hints, not evaluated C++ state.",
             "- Bughouse/Supply board rules do not provide a complete synchronized two-board match.", ""]
    runtime = catalog.get("runtime")
    if runtime:
        lines += [f"Runtime handshake advertised **{len(runtime['variant_ids'])}** variant IDs.",
                  f"Engine SHA-256: `{runtime['engine_sha256']}`.", ""]
        if runtime["source_ids_not_advertised"]:
            lines += ["Source IDs absent in this binary: " + ", ".join(runtime["source_ids_not_advertised"]) + ".", ""]
    else:
        lines += ["Runtime was not queried. Build guards may exclude source registrations.", ""]
    lines += ["## Requested games and additional official candidates", "",
              "See [CHESSCOM_RULES.md](CHESSCOM_RULES.md) for rule parity details and primary sources.", "",
              "| Game | Class | Core IDs / candidate IDs | Runtime | Core tests pass / fail | Scope and remaining work |",
              "|---|---|---|---|---|---|"]
    for game in catalog["requested_games"]:
        observed = game.get("runtime_advertised_ids")
        runtime_text = "not queried" if observed is None else (", ".join(observed) or "none")
        lines.append("| " + " | ".join(cell(value) for value in (
            game["name"], game["classification"], ", ".join(game["core_ids"]) or "—",
            runtime_text,
            str(sum(entry.get("passed") is True for entry in game["validation_evidence"])) + " / "
            + str(sum(entry.get("passed") is False for entry in game["validation_evidence"])),
            game["notes"])) + " |")
    lines += ["", "## Compiled registrations", "",
              "The factory expressions in JSON preserve assigned C++ text and inherited assignments. "
              "Header defaults, control flow and `Variant::conclude()` are not interpreted. "
              "Piece operations shown there are factory-local operations.", "",
              "| ID | Geometry hint | Build guards | Factory | Runtime advertised |",
              "|---|---|---|---|---|"]
    for item in catalog["compiled_registrations"]:
        board = item["geometry_hint"]
        observation = item.get("runtime_advertised")
        lines.append("| " + " | ".join(cell(value) for value in (
            item["id"], f"{board['files']} × {board['ranks']}",
            ", ".join(item["build_guards"]) or "base", item["factory"],
            "not queried" if observation is None else str(observation).lower())) + " |")
    lines += ["", "## INI examples", "",
              "INI entries below are definitions found in `src/variants.ini`, not a claim that each "
              "entry was loaded, accepted, searched, or tested. Unsupported option combinations must "
              "be rejected by the rule loader. JSON contains declared properties and inherited keys.", "",
              "| ID | Parent | Geometry hint | Declared rule keys |",
              "|---|---|---|---|"]
    for item in catalog["ini_examples"]:
        board = item["geometry_hint"]
        lines.append("| " + " | ".join(cell(value) for value in (
            item["id"], item["parent"] or "default template", f"{board['files']} × {board['ranks']}",
            ", ".join(item["properties"]))) + " |")
    lines += ["", "## Validation evidence", "",
              "Runtime listing is not perft, rule parity, search strength, or release validation. "
              "External reports are attached in JSON without changing approval status. The table's "
              "counts concern named core rulesets, not automatic parity with an adapted game. "
              "Zero tests means no attached evidence, not a failed game. Reports from another binary "
              "are rejected when --engine is provided.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "upstream/fairy-stockfish")
    parser.add_argument("--classification", type=Path, default=ROOT / "docs/variant-classification.json")
    parser.add_argument("--json", type=Path, default=ROOT / "docs/variants.json")
    parser.add_argument("--markdown", type=Path, default=ROOT / "docs/VARIANTS.md")
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--load-config", type=Path, action="append", default=[])
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--validation", type=Path, action="append", default=[],
                        help="Attach test reports as evidence; never auto-approve games")
    parser.add_argument("--require-all-builtins", action="store_true",
                        help="Fail if the runtime omits any source registration (all=yes largeboards=yes)")
    args = parser.parse_args()
    if args.load_config and not args.engine:
        parser.error("--load-config requires --engine")
    if args.require_all_builtins and not args.engine:
        parser.error("--require-all-builtins requires --engine")
    try:
        cpp = args.source / "src/variant.cpp"
        ini = args.source / "src/variants.ini"
        header = args.source / "src/variant.h"
        defaults_text = header.read_text(encoding="utf-8")
        rank = re.search(r"\bmaxRank\s*=\s*RANK_(\d+)", defaults_text)
        file = re.search(r"\bmaxFile\s*=\s*FILE_([A-L])", defaults_text)
        if not rank or not file:
            raise ValueError("Cannot identify default board dimensions in variant.h")
        defaults = {"ranks": int(rank.group(1)), "files": ord(file.group(1)) - ord("A") + 1}
        cpp_text = cpp.read_text(encoding="utf-8")
        functions = source_functions(cpp_text)
        builtins = registrations(cpp_text, functions, defaults)
        configs = ini_examples(ini.read_text(encoding="utf-8"), builtins, defaults)
        classification = json.loads(args.classification.read_text(encoding="utf-8"))
        games = classification["games"]
        if len({game["id"] for game in games}) != len(games):
            raise ValueError("Duplicate requested game ID")
        allowed = {"Native", "Adapted", "New Rules", "Special Architecture"}
        if any(game["classification"] not in allowed for game in games):
            raise ValueError("Unknown support classification")
        source_ids = {item["id"] for item in builtins}
        runtime = runtime_inventory(args.engine, args.load_config, args.timeout) if args.engine else None
        observed = set(runtime["variant_ids"]) if runtime else set()
        if runtime:
            runtime["source_ids_not_advertised"] = sorted(source_ids - observed)
            runtime["additional_runtime_ids"] = sorted(observed - source_ids)
            if args.require_all_builtins and runtime["source_ids_not_advertised"]:
                raise ValueError("Missing runtime builtins: " + ", ".join(runtime["source_ids_not_advertised"]))
        evidence, by_variant = validation_reports(args.validation, runtime)
        for item in builtins:
            item["runtime_advertised"] = item["id"] in observed if runtime else None
            item["validation_evidence"] = by_variant.get(item["id"], [])
        for item in configs:
            item["runtime_advertised"] = item["id"] in observed if runtime else None
            item["validation_evidence"] = by_variant.get(item["id"], [])
        for game in games:
            game["runtime_advertised_ids"] = sorted(observed.intersection(game["core_ids"])) if runtime else None
            game["source_registration_ids"] = sorted(source_ids.intersection(game["core_ids"]))
            game["validation_evidence"] = game.get("validation_evidence", []) + [
                entry for variant in game["core_ids"] for entry in by_variant.get(variant, [])]
            game["release_approved"] = False
        catalog = {
            "schema_version": 1,
            "upstream_commit": source_revision(args.source),
            "source_sha256": {"src/variant.cpp": digest(cpp), "src/variant.h": digest(header),
                              "src/variants.ini": digest(ini)},
            "classification_definitions": classification["definitions"],
            "architecture": {"players": 2, "maximum_files": 12, "maximum_ranks": 10,
                             "largeboards_required_above_8x8": True,
                             "full_multi_board_coordinator": False,
                             "observation_only_fog_search": False},
            "extraction_limitations": [
                "C++ assignments are source expressions, not runtime rule introspection",
                "Control flow, header defaults other than geometry, and conclude() are not evaluated",
                "INI examples are neither loaded nor rule-validated by source extraction",
                "Native/source registration, runtime inventory, validation, and release approval differ",
            ],
            "compiled_registrations": builtins,
            "ini_examples": configs,
            "requested_games": games,
            "runtime": runtime,
            "validation_evidence": evidence,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        args.markdown.write_text(markdown(catalog), encoding="utf-8")
        print(json.dumps({"compiled_registrations": len(builtins), "ini_examples": len(configs),
                          "requested_games": len(games),
                          "runtime_variants": len(observed) if runtime else None,
                          "json": str(args.json), "markdown": str(args.markdown)}))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print("catalog error: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
