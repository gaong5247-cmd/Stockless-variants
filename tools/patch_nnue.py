#!/usr/bin/env python3
"""Wire the independent SVNNUE backend into a fully patched, copied FSF tree.

Applied after patch_protocol/patch_rules/patch_pocket_keys; never edits upstream.
"""
from pathlib import Path


def replace(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path.name}: NNUE patch expected one anchor, found {count}: {old[:90]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def apply(out: Path) -> None:
    src = out / "src"
    position = src / "position.h"
    replace(position, '#include "nnue/nnue_accumulator.h"\n',
            '#include "nnue/nnue_accumulator.h"\n#include "stockless_core/nnue.h"\n#include "stockless_core/v3.h"\n')
    replace(position, '  StateInfo* state() const;\n', '''  StateInfo* state() const;
  Stockless::Neural::Accumulator& stockless_nnue_accumulator() const { return stocklessNnue; }
  Stockless::Neural::V3::Accumulator& stockless_v3_accumulator() const { return stocklessV3; }
''')
    replace(position, '  // Data members\n  Piece board[SQUARE_NB];\n', '''  // Data members
  mutable Stockless::Neural::Accumulator stocklessNnue;
  mutable Stockless::Neural::V3::Accumulator stocklessV3;
  Piece board[SQUARE_NB];
''')
    replace(position, '''inline void Position::put_piece(Piece pc, Square s, bool isPromoted, Piece unpromotedPc) {

  board[s] = pc;
''', '''inline void Position::put_piece(Piece pc, Square s, bool isPromoted, Piece unpromotedPc) {

  if (stocklessNnue.generation) Stockless::Neural::board_delta(*this, pc, s, 1);
  if (stocklessV3.generation) Stockless::Neural::V3::board_delta(*this, pc, s, 1);
  board[s] = pc;
''')
    replace(position, '''inline void Position::remove_piece(Square s) {

  Piece pc = board[s];
''', '''inline void Position::remove_piece(Square s) {

  Piece pc = board[s];
  if (stocklessNnue.generation) Stockless::Neural::board_delta(*this, pc, s, -1);
  if (stocklessV3.generation) Stockless::Neural::V3::board_delta(*this, pc, s, -1);
''')
    replace(position, '''inline void Position::move_piece(Square from, Square to) {

  Piece pc = board[from];
''', '''inline void Position::move_piece(Square from, Square to) {

  Piece pc = board[from];
  if (stocklessNnue.generation) {
      Stockless::Neural::board_delta(*this, pc, from, -1);
      Stockless::Neural::board_delta(*this, pc, to, 1);
  }
  if (stocklessV3.generation) {
      Stockless::Neural::V3::board_delta(*this, pc, from, -1);
      Stockless::Neural::V3::board_delta(*this, pc, to, 1);
  }
''')
    replace(position, '''inline void Position::add_to_hand(Piece pc) {
  if (variant()->freeDrops) return;
''', '''inline void Position::add_to_hand(Piece pc) {
  if (variant()->freeDrops) return;
  if (stocklessNnue.generation) Stockless::Neural::pocket_delta(*this, pc, 1);
  if (stocklessV3.generation) Stockless::Neural::V3::pocket_delta(*this, pc, 1);
''')
    replace(position, '''inline void Position::remove_from_hand(Piece pc) {
  if (variant()->freeDrops) return;
''', '''inline void Position::remove_from_hand(Piece pc) {
  if (variant()->freeDrops) return;
  if (stocklessNnue.generation) Stockless::Neural::pocket_delta(*this, pc, -1);
  if (stocklessV3.generation) Stockless::Neural::V3::pocket_delta(*this, pc, -1);
''')

    makefile = src / "Makefile"
    replace(makefile, 'OBJS = $(notdir $(SRCS:.cpp=.o))\n',
            'SRCS += stockless_core/nnue.cpp stockless_core/v3.cpp\n\nOBJS = $(notdir $(SRCS:.cpp=.o))\n')
    replace(makefile, 'VPATH = syzygy:nnue:nnue/features\n',
            'VPATH = syzygy:nnue:nnue/features:stockless_core\n')
    for name in ("evaluate.cpp", "uci.cpp", "ucioption.cpp"):
        replace(src / name, '#include "evaluate.h"\n',
                '#include "evaluate.h"\n#include "stockless_core/nnue.h"\n#include "stockless_core/v3.h"\n')

    evaluation = src / "evaluate.cpp"
    replace(evaluation, '''  if (!Eval::useNNUE || !pos.nnue_applicable())
      v = Evaluation<NO_TRACE>(pos).value();
''', '''  if (Stockless::Neural::V3::applicable(pos))
      v = Stockless::Neural::V3::evaluate(pos);
  else if (Stockless::Neural::applicable(pos))
      v = Stockless::Neural::evaluate(pos);
  else if (!Eval::useNNUE || !pos.nnue_applicable())
      v = Evaluation<NO_TRACE>(pos).value();
''')
    replace(evaluation, '''        else
            sync_cout << "info string classical evaluation enabled" << sync_endl;
''', '''        else if (std::string(Stockless::Neural::backend_name()) != "Classical")
            sync_cout << "info string NNUE_STATUS "
                      << Stockless::Neural::status_json(variants.find(Options["UCI_Variant"])->second) << sync_endl;
        else
            sync_cout << "info string classical evaluation enabled" << sync_endl;
''')
    replace(evaluation, '''  ss << "\\nClassical evaluation   " << to_cp(v) << " (white side)\\n";
''', '''  ss << "\\nClassical evaluation   " << to_cp(v) << " (white side)\\n";
  if (Stockless::Neural::applicable(pos)) {
      const auto neural = Stockless::Neural::probe(pos, false);
      ss << "StocklessNNUE SVNNUE-v1 " << double(neural.white_cp) / 100.0 << " (white side)\\n";
  }
''')
    replace(evaluation, '''  ss << "Final evaluation       " << to_cp(v) << " (white side)";
''', '''  ss << "Final evaluation       " << to_cp(v) << " (white side)";
  if (Stockless::Neural::applicable(pos))
      ss << " [StocklessNNUE; rule draw scaling and value bounds applied]";
''')

    option = src / "ucioption.cpp"
    # set_default() invokes callbacks in upstream. Restoring a rejected path
    # with it would recursively retry the same invalid/default file forever.
    replace(src / "uci.h", '  void set_default(std::string newDefault);\n',
            '  void set_default(std::string newDefault);\n  void set_current_without_callback(const std::string& value);\n')
    replace(option, 'const std::string Option::get_type() const {\n', '''void Option::set_current_without_callback(const std::string& value) {
    currentValue = value;
}

const std::string Option::get_type() const {
''')
    replace(option, '''void on_eval_file(const Option&) {
    sync_cout << "info string ERROR EvalFile loading is disabled until variant network validation is implemented" << sync_endl;
}
void on_eval_backend(const Option&) {
    Options["Use NNUE"] = std::string("false");
}
''', '''void stop_neural_search() {
    if (!Threads.empty()) {
        Threads.stop = true;
        Threads.main()->wait_for_search_finished();
    }
}
void on_eval_file(const Option& o) {
    stop_neural_search();
    std::string error;
    if (!Stockless::Neural::load_network(std::string(o), variants.find(Options["UCI_Variant"])->second, error)) {
        Options["EvalFile"].set_current_without_callback(Stockless::Neural::loaded_file());
        sync_cout << "info string ERROR " << error << sync_endl;
        return;
    }
    if (!Threads.empty()) Search::clear();
    sync_cout << "info string NNUE_STATUS "
              << Stockless::Neural::status_json(variants.find(Options["UCI_Variant"])->second) << sync_endl;
}
void on_eval_backend(const Option& o) {
    stop_neural_search();
    std::string error;
    if (!Stockless::Neural::select_backend(std::string(o), variants.find(Options["UCI_Variant"])->second, error)) {
        Options["EvalBackend"].set_current_without_callback(Stockless::Neural::backend_name());
        sync_cout << "info string ERROR " << error << sync_endl;
        return;
    }
    Options["Use NNUE"] = std::string("false");
    if (!Threads.empty()) Search::clear();
    sync_cout << "info string NNUE_STATUS "
              << Stockless::Neural::status_json(variants.find(Options["UCI_Variant"])->second) << sync_endl;
}
''')
    replace(option, 'info string ERROR NNUE unavailable: a validated variant network manifest is required',
            'info string ERROR Use NNUE selects the unavailable FSF network format; use EvalBackend StocklessNNUE with a validated SVNNUE EvalFile')
    replace(option, 'std::vector<std::string>{"Auto", "Classical"}, on_eval_backend',
            'std::vector<std::string>{"Auto", "Classical", "StocklessNNUE", "StocklessNNUEv3"}, on_eval_backend')
    replace(option, '''    PSQT::init(v);
    // Position/material/pawn keys''', '''    PSQT::init(v);
    const std::string oldBackend = Stockless::Neural::backend_name();
    Stockless::Neural::variant_changed(v);
    if (Options.count("EvalBackend")) Options["EvalBackend"].set_current_without_callback(Stockless::Neural::backend_name());
    if (oldBackend != Stockless::Neural::backend_name())
        sync_cout << "info string StocklessNNUE disabled: selected variant needs its own compatible network" << sync_endl;
    // Position/material/pawn keys''')

    uci = src / "uci.cpp"
    replace(uci, '          "load", "check", "flip", "bench", "eval", "export_net"\n',
            '          "load", "check", "flip", "bench", "eval", "export_net", "nnueeval", "nnuestatus"\n')
    replace(uci, '''      else if (CurrentProtocol == XBOARD)
          XBoard::stateMachine->process_command(token, is);
''', '''      else if (token == "nnueeval" || token == "nnuestatus") {
          Threads.stop = true;
          Threads.main()->wait_for_search_finished();
          if (token == "nnueeval")
              sync_cout << "info string NNUE_EVAL " << Stockless::Neural::evaluation_json(pos) << sync_endl;
          else
              sync_cout << "info string NNUE_STATUS " << Stockless::Neural::status_json(pos.variant()) << sync_endl;
      }
      else if (CurrentProtocol == XBOARD)
          XBoard::stateMachine->process_command(token, is);
''')
