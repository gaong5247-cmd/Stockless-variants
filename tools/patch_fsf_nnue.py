#!/usr/bin/env python3
"""Restore the pinned Fairy-Stockfish .nnue reader in the generated overlay.

This runs after patch_nnue.py.  FSF reads .nnue bytes without conversion.
SVNNUE-v1 remains an opt-in separate backend, selected by a single .svnnue.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CODE = r'''// Installed packages may ship several unmodified FSF variant networks.
// Names are generated from the official-download catalog at build time.
std::string fsf_bundle_eval_files() {
    const char* names[] = {
@NAMES@
    };
    std::string value;
    for (const char* name : names) {
        if (!value.empty()) value += SepChar;
        value += "networks/";
        value += name;
    }
    return value;
}
bool fsf_bundle_available() {
    const std::string file = "networks/@CHESS@";
    return std::ifstream(file, std::ios::binary).good()
        || std::ifstream(CommandLine::binaryDirectory + file, std::ios::binary).good();
}
bool fsf_network_exists_for_variant() {
    const std::string variant = std::string(Options["UCI_Variant"]);
    const std::string alias = variants.find(variant)->second->nnueAlias;
    std::istringstream paths{std::string(Options["EvalFile"])};
    std::string path;
    while (std::getline(paths, path, SepChar)) {
        const std::string basename = path.substr(path.find_last_of("\\/") + 1);
        if (basename.rfind(variant, 0) != 0
            && (alias.empty() || basename.rfind(alias, 0) != 0))
            continue;
        if (std::ifstream(path, std::ios::binary).good()
            || std::ifstream(CommandLine::binaryDirectory + path, std::ios::binary).good())
            return true;
    }
    return false;
}
void stop_neural_search() {
    if (!Threads.empty()) {
        Threads.stop = true;
        Threads.main()->wait_for_search_finished();
    }
}
void on_use_NNUE(const Option& o) {
    stop_neural_search();
    if (double(o) && !fsf_network_exists_for_variant()) {
        Options["Use NNUE"].set_current_without_callback("false");
        Eval::NNUE::init();
        sync_cout << "info string ERROR no readable FSF .nnue file matches this variant in EvalFile" << sync_endl;
        return;
    }
    Eval::NNUE::init();
    if (!Threads.empty()) Search::clear();
}
void on_eval_file(const Option& o) {
    stop_neural_search();
    const std::string path = std::string(o);
    // Keep the existing SVNNUE-v1 file loader, which has a different format.
    if (path.size() >= 7 && path.compare(path.size() - 7, 7, ".svnnue") == 0) {
        Options["Use NNUE"].set_current_without_callback("false");
        Eval::NNUE::init();
        std::string error;
        if (!Stockless::Neural::load_network(path, variants.find(Options["UCI_Variant"])->second, error)) {
            Options["EvalFile"].set_current_without_callback(Stockless::Neural::loaded_file());
            sync_cout << "info string ERROR " << error << sync_endl;
            return;
        }
        sync_cout << "info string NNUE_STATUS "
                  << Stockless::Neural::status_json(variants.find(Options["UCI_Variant"])->second) << sync_endl;
    } else {
        std::string unused;
        Stockless::Neural::select_backend("Classical", variants.find(Options["UCI_Variant"])->second, unused);
        Options["EvalBackend"].set_current_without_callback("Classical");
        if (Options["Use NNUE"] && !fsf_network_exists_for_variant()) {
            Options["EvalFile"].set_current_without_callback(fsf_bundle_eval_files());
            Options["Use NNUE"].set_current_without_callback("false");
            sync_cout << "info string ERROR FSF EvalFile is not readable for the current variant" << sync_endl;
        }
        Eval::NNUE::init();
    }
    if (!Threads.empty()) Search::clear();
}
void on_eval_backend(const Option& o) {
    stop_neural_search();
    std::string error;
    if (!Stockless::Neural::select_backend(std::string(o), variants.find(Options["UCI_Variant"])->second, error)) {
        Options["EvalBackend"].set_current_without_callback(Stockless::Neural::backend_name());
        sync_cout << "info string ERROR " << error << sync_endl;
        return;
    }
    // A deliberate experimental/classical backend selection disables upstream
    // NNUE. 'Use NNUE true' switches back to FSF when a matching file exists.
    Options["Use NNUE"] = std::string("false");
    if (!Threads.empty()) Search::clear();
    sync_cout << "info string NNUE_STATUS "
              << Stockless::Neural::status_json(variants.find(Options["UCI_Variant"])->second) << sync_endl;
}

'''


def replace_once(path: Path, old: str, new: str):
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError(f"FSF NNUE integration anchor changed: {path.name}: {old[:70]}")
    path.write_text(source.replace(old, new, 1), encoding="utf-8")


def apply(out: Path):
    catalog = json.loads((ROOT / "config/fsf-networks.json").read_text(encoding="utf-8"))
    names = ",\n".join('        "' + item["filename"] + '"' for item in catalog["networks"])
    chess = next(item["filename"] for item in catalog["networks"] if item["variant"] == "chess")
    handlers = CODE.replace("@NAMES@", names).replace("@CHESS@", chess)
    option = out / "src" / "ucioption.cpp"
    source = option.read_text(encoding="utf-8")
    start = 'void on_use_NNUE(const Option& o) {'
    end = 'void on_variant_path(const Option& o) {'
    if source.count(start) != 1 or source.count(end) != 1 or source.index(start) >= source.index(end):
        raise RuntimeError("Patched UCI callback boundaries changed")
    source = source[:source.index(start)] + handlers + source[source.index(end):]
    option.write_text(source, encoding="utf-8")

    replace_once(option, '  o["Use NNUE"]              << Option(false, on_use_NNUE);',
                 '  o["Use NNUE"]              << Option(fsf_bundle_available(), on_use_NNUE);')
    replace_once(option, '  o["EvalFile"]              << Option("<empty>", on_eval_file);',
                 '  o["EvalFile"]              << Option(fsf_bundle_eval_files().c_str(), on_eval_file);')
