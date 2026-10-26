#!/usr/bin/env python3
"""Checked protocol/configuration patches for the pinned Fairy-Stockfish tree.

Stockless Variants contributors, 2026. SPDX-License-Identifier: AGPL-3.0-only
Upstream files retain their GPL-3.0-or-later notices. This module is called by
materialize.py, not run against the read-only upstream checkout.
"""
from pathlib import Path


def replace(path: Path, old: str, new: str) -> None:
    data = path.read_text(encoding="utf-8")
    count = data.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one protocol patch anchor, got {count}: {old[:70]!r}")
    path.write_text(data.replace(old, new, 1), encoding="utf-8")


def span(path: Path, start: str, end: str, new: str) -> None:
    data = path.read_text(encoding="utf-8")
    if data.count(start) != 1 or data.count(end) != 1:
        raise RuntimeError(f"{path}: protocol patch span changed")
    begin, finish = data.index(start), data.index(end)
    if finish <= begin:
        raise RuntimeError(f"{path}: invalid protocol patch order")
    path.write_text(data[:begin] + new + data[finish:], encoding="utf-8")


GUARD = r'''// Stockless Variants protocol/configuration guards.
// Copyright (C) 2026 Stockless Variants contributors
// SPDX-License-Identifier: AGPL-3.0-only
#pragma once
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "../apiutil.h"
#include "../variant.h"

namespace Stockfish::Stockless {
inline int ProtocolExitCode = 0;

inline std::string trim_text(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}

inline bool config_error(const std::string& message) {
    ProtocolExitCode = 2;
    std::cerr << "ERROR variant configuration: " << message << std::endl;
    return false;
}

// This is deliberately only the grammar understood by the pinned FSF Betza
// parser. Unknown atoms/modifiers must not silently turn into different rules.
inline bool valid_betza(const std::string& expression) {
    const std::string atoms = "WFDNAHLCJZGKRBQ";
    const std::string modifiers = "mcpgni fbvhrlse";
    bool sawAtom = false, pendingModifier = false;
    for (std::size_t i = 0; i < expression.size(); ++i) {
        const char c = expression[i];
        if (atoms.find(c) != std::string::npos) {
            sawAtom = true;
            pendingModifier = false;
            if (i + 1 < expression.size()
                && (expression[i + 1] == c
                    || (expression[i + 1] >= '1' && expression[i + 1] <= '9')))
                ++i;
        } else if (c != ' ' && modifiers.find(c) != std::string::npos) {
            pendingModifier = true;
        } else return false;
    }
    return sawAtom && !pendingModifier;
}

// Normalize whitespace before the upstream parser and reject skipped syntax.
// Repeated keys retain FSF's documented last-assignment semantics.
inline bool config_lex(const std::string& source, std::string& normalized) {
    if (source.size() > 1024 * 1024)
        return config_error("configuration exceeds the 1 MiB limit");
    std::istringstream input(source);
    std::ostringstream output;
    std::set<std::string> sections, keys;
    bool inSection = false;
    std::string line;
    unsigned lineNumber = 0;
    while (std::getline(input, line)) {
        ++lineNumber;
        line = trim_text(line);
        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
        if (line[0] == '[') {
            if (line.size() < 3 || line.back() != ']'
                || line.find(']') != line.size() - 1)
                return config_error("invalid section header at line " + std::to_string(lineNumber));
            std::string name = line.substr(1, line.size() - 2);
            const auto colon = name.find(':');
            if (name.empty() || name.find_first_of(" \t[]") != std::string::npos
                || (colon != std::string::npos
                    && (colon == 0 || colon + 1 == name.size() || name.find(':', colon + 1) != std::string::npos)))
                return config_error("invalid variant/template name at line " + std::to_string(lineNumber));
            name = name.substr(0, colon);
            if (!sections.insert(name).second)
                return config_error("duplicate variant: " + name);
            keys.clear();
            inSection = true;
            output << line << '\n';
            continue;
        }
        const auto equals = line.find('=');
        if (!inSection || equals == std::string::npos)
            return config_error("invalid syntax at line " + std::to_string(lineNumber));
        std::string key = trim_text(line.substr(0, equals));
        std::string value = trim_text(line.substr(equals + 1));
        if (key.empty() || value.empty() || key.find_first_of(" \t[]") != std::string::npos)
            return config_error("empty or invalid key/value at line " + std::to_string(lineNumber));
        if ((key.rfind("customPiece", 0) == 0 || key == "king") && value.size() > 1) {
            if (value.size() < 3 || value[1] != ':' || !valid_betza(value.substr(2)))
                return config_error("unsupported Betza notation for " + key + ": " + value);
        }
        output << key << " = " << value << '\n';
    }
    if (!inSection) return config_error("no variant sections found");
    normalized = output.str();
    return true;
}

// The checked upstream pass uses temporary registry entries for inheritance and
// removes ALL of them before returning. Capture diagnostics to turn errors into
// a transaction failure instead of proceeding with a partly valid registry.
inline bool validate_variant_text(const std::string& source) {
    std::string normalized;
    if (!config_lex(source, normalized)) return false;
    std::istringstream checked(normalized);
    std::ostringstream diagnostics;
    auto* previous = std::cerr.rdbuf(diagnostics.rdbuf());
    variants.parse_istream<true>(checked);
    std::cerr.rdbuf(previous);
    bool valid = true;
    std::istringstream messages(diagnostics.str());
    std::string line;
    while (std::getline(messages, line)) {
        if (line.empty() || line.rfind("Parsing variant: ", 0) == 0) continue;
        if (line.find("Deprecated option") != std::string::npos)
            std::cerr << "WARNING variant configuration: " << line << std::endl;
        else {
            std::cerr << "ERROR variant configuration: " << line << std::endl;
            valid = false;
        }
    }
    if (!valid) ProtocolExitCode = 2;
    return valid;
}

inline bool check_variant_path(const std::string& path) {
    std::ifstream input(path);
    if (!input) return config_error("unable to open file " + path);
    const std::string source((std::istreambuf_iterator<char>(input)), {});
    return validate_variant_text(source);
}

inline bool small_unsigned(const std::string& value, unsigned& result, unsigned maximum) {
    if (value.empty()) return false;
    result = 0;
    for (unsigned char c : value) {
        if (!std::isdigit(c) || result > maximum / 10) return false;
        result = result * 10 + (c - '0');
        if (result > maximum) return false;
    }
    return true;
}

inline bool valid_position_text(const std::string& text, const Variant* variant,
                                bool chess960, bool sfen = false) {
    if (text.empty() || text.size() > 16384) return false;
    std::istringstream input(text);
    std::vector<std::string> fields;
    std::string field;
    while (input >> field) fields.push_back(field);
    if (fields.empty()) return false;
    std::string fen;
    if (sfen) {
        unsigned moveNumber;
        if (fields.size() != 4 || (fields[1] != "b" && fields[1] != "w")
            || !small_unsigned(fields[3], moveNumber, 1000000) || !moveNumber)
            return false;
        std::string pocket;
        if (fields[2] != "-") {
            unsigned count = 0;
            for (unsigned char c : fields[2]) {
                if (std::isdigit(c)) {
                    count = count * 10 + (c - '0');
                    if (count >= SQUARE_NB) return false;
                } else {
                    if (!std::isalpha(c) || variant->pieceToChar.find(char(c)) == std::string::npos)
                        return false;
                    pocket.append(count ? count : 1, char(c));
                    count = 0;
                    if (pocket.size() >= SQUARE_NB) return false;
                }
            }
            if (count) return false;
        }
        fen = fields[0] + "[" + pocket + "] " + (fields[1] == "b" ? "w" : "b") + " - - 0 1";
    } else {
        for (const auto& part : fields) fen += (fen.empty() ? "" : " ") + part;
        if (fields.size() > 1 && fields[1] != "w" && fields[1] != "b") return false;
        for (std::size_t i = 2; i < fields.size(); ++i)
            if (std::all_of(fields[i].begin(), fields[i].end(), [](unsigned char c) { return std::isdigit(c); })) {
                unsigned number;
                if (!small_unsigned(fields[i], number, 1000000)) return false;
            }
    }
    return FEN::validate_fen(fen, variant, chess960) == FEN::FEN_OK;
}
} // namespace Stockfish::Stockless
'''


POSITION = r'''  void position(Position& pos, istringstream& is, StateListPtr& states) {
    string token, fen;
    is >> token;
    const bool sfen = token == "sfen";
    const Variant* variant = variants.find(Options["UCI_Variant"])->second;
    if (token == "startpos") {
        fen = variant->startFen;
        if ((is >> token) && token != "moves") {
            sync_cout << "info string ERROR expected moves after startpos" << sync_endl;
            return;
        }
    } else if (token == "fen" || token == "sfen") {
        while (is >> token && token != "moves") fen += token + " ";
    } else {
        sync_cout << "info string ERROR invalid position command" << sync_endl;
        return;
    }
    fen = Stockless::trim_text(fen);
    if (!Stockless::valid_position_text(fen, variant, Options["UCI_Chess960"], sfen)) {
        sync_cout << "info string ERROR invalid " << (sfen ? "SFEN" : "FEN") << sync_endl;
        return;
    }
    // Validate the complete command before discarding the previous position.
    Position candidate;
    std::deque<StateInfo> candidateStates(1);
    candidate.set(variant, fen, Options["UCI_Chess960"], &candidateStates.back(), Threads.main(), sfen);
    std::vector<Move> moves;
    while (is >> token) {
        Move move = UCI::to_move(candidate, token);
        if (move == MOVE_NONE) {
            sync_cout << "info string ERROR illegal move: " << token << sync_endl;
            return;
        }
        moves.push_back(move);
        candidateStates.emplace_back();
        candidate.do_move(move, candidateStates.back());
    }
    states = StateListPtr(new std::deque<StateInfo>(1));
    pos.set(variant, fen, Options["UCI_Chess960"], &states->back(), Threads.main(), sfen);
    for (Move move : moves) {
        states->emplace_back();
        pos.do_move(move, states->back());
    }
  }

'''


def apply(out: Path) -> None:
    out = Path(out)
    src = out / "src"
    core = src / "stockless_core"
    core.mkdir(exist_ok=True)
    (core / "protocol_guard.h").write_text(GUARD, encoding="utf-8")
    for name in ("variant.cpp", "uci.cpp", "ucioption.cpp", "main.cpp", "xboard.cpp"):
        path = src / name
        anchor = '#include "uci.h"\n' if name != "variant.cpp" else '#include "variant.h"\n'
        replace(path, anchor, anchor + '#include "stockless_core/protocol_guard.h"\n')
    replace(src / "ucioption.cpp", '#include "stockless_core/protocol_guard.h"\n',
            '#include "stockless_core/protocol_guard.h"\n#include "stockless_core/config.h"\n#include <cmath>\n')
    replace(src / "main.cpp", '  return 0;\n', '  return Stockless::ProtocolExitCode;\n')

    # All normal config entry points, including env/VariantPath and here-docs,
    # validate the entire source before they can register the first definition.
    replace(src / "variant.cpp", 'void VariantMap::parse_istream(std::istream& file) {\n',
            '''void VariantMap::parse_istream(std::istream& sourceInput) {
    const std::string source((std::istreambuf_iterator<char>(sourceInput)), {});
    std::string normalized;
    if (!Stockless::config_lex(source, normalized)) return;
    if (!DoCheck && !Stockless::validate_variant_text(normalized)) return;
    std::istringstream file(normalized);
''')
    replace(src / "variant.cpp", '        std::cerr << "Unable to open file " << path << std::endl;\n',
            '        Stockless::config_error("unable to open file " + path);\n')
    replace(src / "uci.cpp", '            variants.parse_istream<true>(ss);\n',
            '            Stockless::validate_variant_text(ss.str());\n')
    replace(src / "uci.cpp", '                variants.parse<true>(token.erase(end + 1));\n',
            '                Stockless::check_variant_path(token.erase(end + 1));\n')
    replace(src / "uci.cpp", '''                Options["VariantPath"] = token.erase(end + 1);
        }
    }
  }
''', '''                Options["VariantPath"] = token.erase(end + 1);
        }
        else
            Stockless::config_error("missing configuration path or here-doc");
    }
  }
''')

    # Prevent malformed numerics from mutating enum geometry or invoking stoi.
    replace(src / "parser.cpp", '''        ss >> target;
        return !ss.fail();
''', '''        T parsed{};
        if (!(ss >> parsed)) return false;
        ss >> std::ws;
        if (!ss.eof()) return false;
        target = parsed;
        return true;
''')
    span(src / "parser.cpp", '    template <> bool set(const std::string& value, Rank& target) {',
         '    template <> bool set(const std::string& value, std::string& target) {', r'''    template <> bool set(const std::string& value, Rank& target) {
        std::istringstream input(value);
        int number;
        if (!(input >> number) || number < 1 || number > RANK_MAX + 1) return false;
        input >> std::ws;
        if (!input.eof()) return false;
        target = Rank(number - 1);
        return true;
    }

    template <> bool set(const std::string& value, File& target) {
        int number;
        if (value.size() == 1 && value[0] >= 'a' && value[0] <= 'a' + FILE_MAX)
            number = value[0] - 'a' + 1;
        else {
            std::istringstream input(value);
            if (!(input >> number)) return false;
            input >> std::ws;
            if (!input.eof()) return false;
        }
        if (number < 1 || number > FILE_MAX + 1) return false;
        target = File(number - 1);
        return true;
    }

''')
    span(src / "parser.cpp", '    template <> bool set(const std::string& value, Bitboard& target) {',
         '    template <> bool set(const std::string& value, CastlingRights& target) {', r'''    template <> bool set(const std::string& value, Bitboard& target) {
        std::istringstream input(value);
        std::string token;
        Bitboard parsed = 0;
        while (input >> token) {
            if (token == "-") continue;
            if (token.size() < 2) return false;
            if (token.back() == '*' && token.size() == 2) {
                const char file = char(tolower(token[0]));
                if (file < 'a' || file > 'a' + FILE_MAX) return false;
                parsed |= file_bb(File(file - 'a'));
                continue;
            }
            std::istringstream numberInput(token.substr(1));
            int rank;
            if (!(numberInput >> rank) || rank < 1 || rank > RANK_MAX + 1) return false;
            numberInput >> std::ws;
            if (!numberInput.eof()) return false;
            if (token[0] == '*') parsed |= rank_bb(Rank(rank - 1));
            else {
                const char file = char(tolower(token[0]));
                if (file < 'a' || file > 'a' + FILE_MAX) return false;
                parsed |= square_bb(make_square(File(file - 'a'), Rank(rank - 1)));
            }
        }
        target = parsed;
        return true;
    }

''')
    replace(src / "parser.cpp", 'return value == "makruk" || value == "asean" || value == "none";',
            'return value == "makruk" || value == "cambodian" || value == "asean" || value == "none";')

    # Upstream FEN utility misses the last rank width and can index past the
    # board after an oversized empty-square run; validate before Position::set.
    replace(src / "apiutil.h", '            if (fileIdx == board.get_nb_files())\n',
            '            if (fileIdx >= board.get_nb_files() || rankIdx >= board.get_nb_ranks())\n')
    replace(src / "apiutil.h", '''        prevChar = c;
    }

    if (v->pieceDrops)
''', '''        if (fileIdx > board.get_nb_files()) {
            std::cerr << "Invalid rank width." << std::endl;
            return NOK;
        }
        prevChar = c;
    }
    if (rankIdx < board.get_nb_ranks() && fileIdx != board.get_nb_files()) {
        std::cerr << "Invalid final rank width." << std::endl;
        return NOK;
    }

    if (v->pieceDrops)
''')

    span(src / "ucioption.cpp", 'void on_use_NNUE(const Option& )', 'void on_variant_set(const Option &o)', r'''void on_use_NNUE(const Option& o) {
    if (double(o)) {
        sync_cout << "info string ERROR NNUE unavailable: a validated variant network manifest is required" << sync_endl;
        Options["Use NNUE"] = std::string("false");
        return;
    }
    Eval::NNUE::init();
}
void on_eval_file(const Option&) {
    sync_cout << "info string ERROR EvalFile loading is disabled until variant network validation is implemented" << sync_endl;
}
void on_eval_backend(const Option&) {
    Options["Use NNUE"] = std::string("false");
}

void on_variant_path(const Option& o) {
    std::stringstream paths((std::string)o), merged;
    std::string path;
    bool hasPath = false;
    while (std::getline(paths, path, SepChar)) {
        path = Stockless::trim_text(path);
        if (path.empty() || path == "<empty>") continue;
        std::ifstream input(path);
        if (!input) { Stockless::config_error("unable to open file " + path); return; }
        merged << input.rdbuf() << '\n';
        hasPath = true;
    }
    if (hasPath) variants.parse_istream<false>(merged);
    Options["UCI_Variant"].set_combo(variants.get_keys());
    if (Options.count("VariantName")) Options["VariantName"].set_combo(variants.get_keys());
    if (Options.count("VariantConfigPath")) Options["VariantConfigPath"].set_default(std::string(o));
}
''')
    replace(src / "ucioption.cpp", '  o["Use NNUE"]              << Option(true, on_use_NNUE);\n',
            '  o["Use NNUE"]              << Option(false, on_use_NNUE);\n')
    anchor = '  o["VariantPath"]           << Option("<empty>", on_variant_path);\n'
    replace(src / "ucioption.cpp", anchor, anchor + '''  o["VariantName"]           << Option("chess", variants.get_keys());
  o["VariantConfigPath"]     << Option("<empty>");
  o["EvalBackend"]           << Option("Classical", std::vector<std::string>{"Auto", "Classical"}, on_eval_backend);
  o["StocklessHybrid"]       << Option(true, [](const Option& value) { Stockless::set_hybrid_enabled(double(value)); });
  o["StocklessAggression"]   << Option(Stockless::default_hybrid_aggression(), 0, 200, [](const Option& value) { Stockless::set_hybrid_aggression(int(double(value))); });
  o["StocklessThreats"]      << Option(true, [](const Option& value) { Stockless::set_threats_enabled(double(value)); });
  o["StocklessOverdrive"]    << Option(Stockless::default_overdrive(), [](const Option& value) { Stockless::set_overdrive_enabled(double(value)); });
  o["StocklessMobile"]       << Option(Stockless::default_mobile_profile(), [](const Option& value) { Stockless::set_mobile_profile(double(value)); });
''')
    replace(src / "ucioption.cpp", '    // Re-initialize NNUE\n',
            '''    if (!Threads.empty()) {
        Threads.stop = true;
        Threads.main()->wait_for_search_finished();
    }
    if (Options.count("VariantName")) Options["VariantName"].set_default(std::string(o));
    // Re-initialize NNUE
''')
    replace(src / "misc.h", '  Entry* operator[](Key key) { return &table[(uint32_t)key & (Size - 1)]; }\n',
            '''  Entry* operator[](Key key) { return &table[(uint32_t)key & (Size - 1)]; }
  void clear() { std::fill(table.begin(), table.end(), Entry{}); }
''')
    replace(src / "misc.h", '#include <cassert>\n', '#include <cassert>\n#include <algorithm>\n')
    replace(src / "ucioption.cpp", '''    PSQT::init(v);
}
void on_variant_change''', '''    PSQT::init(v);
    // Position/material/pawn keys do not encode a custom variant's movement
    // definition. A GUI need not send newgame after changing variants.
    if (!Threads.empty()) {
        Search::clear();
        for (Thread* thread : Threads) {
            thread->pawnsTable.clear();
            thread->materialTable.clear();
        }
    }
}
void on_variant_change''')
    start = '  if (   (type != "button" && v.empty())\n'
    end = '  if (type == "combo")\n  {\n      OptionsMap comboMap;'
    data = (src / "ucioption.cpp").read_text()
    begin, finish = data.index(start), data.index(end)
    data = data[:begin] + r'''  if (Options.count("VariantName") && this == &Options["VariantName"]) {
      Options["UCI_Variant"] = v;
      currentValue = std::string(Options["UCI_Variant"]);
      return *this;
  }
  if (Options.count("VariantConfigPath") && this == &Options["VariantConfigPath"]) {
      Options["VariantPath"] = v;
      currentValue = std::string(Options["VariantPath"]);
      return *this;
  }
  double number = 0;
  bool validNumber = true;
  if (type == "spin") {
      std::istringstream input(v);
      validNumber = bool(input >> number) && std::isfinite(number);
      input >> std::ws;
      validNumber = validNumber && input.eof() && number >= min && number <= max;
  }
  if ((type != "button" && v.empty())
      || (type == "check" && v != "true" && v != "false")
      || (type == "combo" && std::find(comboValues.begin(), comboValues.end(), v) == comboValues.end())
      || !validNumber) {
      sync_cout << "info string ERROR invalid option value: " << v << sync_endl;
      return *this;
  }

''' + data[finish:]
    (src / "ucioption.cpp").write_text(data)

    span(src / "uci.cpp", '  void position(Position& pos, istringstream& is, StateListPtr& states) {',
         '  // trace_eval() prints', POSITION)
    replace(src / "uci.cpp", '      token.clear(); // Avoid a stale if getline() returns empty or blank line\n      is >> skipws >> token;\n', r'''      token.clear(); // Avoid a stale if getline() returns empty or blank line
      is >> skipws >> token;

      // A GUI may replace a position/options during an infinite or ponder
      // search. Join the old workers before mutating tables or StateInfo.
      const std::set<std::string> mutatingCommands = {
          "uci", "usi", "ucci", "ucicyclone", "xboard", "setoption", "position",
          "fen", "startpos", "ucinewgame", "usinewgame", "uccinewgame", "go",
          "load", "check", "flip", "bench", "eval", "export_net"
      };
      if (CurrentProtocol != XBOARD && mutatingCommands.count(token)) {
          Threads.stop = true;
          Threads.main()->wait_for_search_finished();
      }
''')
    replace(src / "uci.cpp", '\n      else if (token == "setoption")  setoption(is);\n', r'''
      else if (token == "setoption") {
          setoption(is);
          if (pos.variant() != variants.find(Options["UCI_Variant"])->second) {
              std::istringstream start("startpos");
              position(pos, start, states);
              banmoves.clear();
          }
      }
''')
    # Native XBoard already stops for most mutations; option was an exception.
    replace(src / "xboard.cpp", '  else if (token == "option")\n  {\n',
            '  else if (token == "option")\n  {\n      stop();\n')
    replace(src / "xboard.cpp", '''      std::getline(is >> std::ws, fen);
      // Check if setboard''', '''      std::getline(is >> std::ws, fen);
      if (!Stockless::valid_position_text(fen, variants.find(Options["UCI_Variant"])->second, Options["UCI_Chess960"])) {
          sync_cout << "Error (invalid FEN)" << sync_endl;
          return;
      }
      // Check if setboard''')
