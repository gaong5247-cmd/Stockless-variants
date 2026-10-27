#!/usr/bin/env python3
"""Download and hash-verify unmodified NNUE files linked by Fairy-Stockfish.

Downloads happen in GitHub Actions (or on an end user's machine), not while
materializing C++ sources. Invalid, incomplete or substituted files fail closed.
Google Drive links require the pinned 'gdown' helper (pip install gdown==5.2.0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "config" / "fsf-networks.json"
MANIFEST = "fsf-networks-manifest.json"


def entries():
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or data.get("source") != "https://fairy-stockfish.github.io/nnue/":
        raise ValueError("Unknown FSF source catalog format")
    variants = set()
    filenames = set()
    for entry in data["networks"]:
        variant, name, url = entry["variant"], entry["filename"], entry["url"]
        if not re.fullmatch(r"[a-z0-9-]+", variant):
            raise ValueError(f"Unsafe variant: {variant!r}")
        if not re.fullmatch(r"[a-z0-9-]+-[0-9a-f]{12}\\.nnue", name):
            raise ValueError(f"Unexpected NNUE filename: {name!r}")
        expected_prefix = "nn-" if variant == "chess" else variant + "-"
        if not name.startswith(expected_prefix):
            raise ValueError(f"Network does not match variant: {name}")
        if variant in variants or name in filenames:
            raise ValueError("Duplicate network or variant in catalog")
        variants.add(variant)
        filenames.add(name)
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in (
                "drive.google.com", "tests.stockfishchess.org"):
            raise ValueError(f"Untrusted NNUE download source: {url}")
    return data


def inspect(path: Path, expected_name: str) -> dict:
    if not path.is_file() or path.stat().st_size < 4096:
        raise ValueError(f"Missing or unexpectedly small NNUE: {path}")
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    digest = sha.hexdigest()
    prefix = expected_name.rsplit("-", 1)[-1].split(".", 1)[0]
    if not digest.startswith(prefix):
        raise ValueError(
            f"NNUE SHA256 does not match its official hashed filename: "
            f"{expected_name}, got {digest}")
    return {"filename": expected_name, "bytes": path.stat().st_size, "sha256": digest}


def fetch_one(entry: dict, destination: Path):
    url = entry["url"]
    if urllib.parse.urlparse(url).hostname == "drive.google.com":
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError("Google Drive downloads require gdown==5.2.0") from exc
        file_id = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["id"][0]
        result = gdown.download(id=file_id, output=str(destination), quiet=False)
        if not result:
            raise RuntimeError(f"FSF network download failed: {entry['filename']}")
    else:
        request = urllib.request.Request(url, headers={
            "User-Agent": "Stockless-variants-FSF-NNUE-bundler/1.0",
        })
        with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as target:
            if response.geturl().startswith("http://"):
                raise RuntimeError("NNUE download redirected to insecure transport")
            shutil.copyfileobj(response, target, length=1024 * 1024)


def download(out: Path):
    data = entries()
    nets = out / "networks"
    nets.mkdir(parents=True, exist_ok=True)
    records = []
    for entry in data["networks"]:
        path = nets / entry["filename"]
        if path.is_file():
            record = inspect(path, entry["filename"])
            print(f"Reusing verified {path.name}", flush=True)
        else:
            temporary = path.with_suffix(".partial")
            temporary.unlink(missing_ok=True)
            try:
                print(f"Downloading official {entry['variant']} NNUE: {entry['filename']}", flush=True)
                fetch_one(entry, temporary)
                record = inspect(temporary, entry["filename"])
                temporary.rename(path)
            finally:
                temporary.unlink(missing_ok=True)
        records.append({**record, "variant": entry["variant"], "source_url": entry["url"],
                        "author": entry.get("author", "")})
    record_manifest = {
        "schema": 1,
        "source_catalog": data["source"],
        "files": records,
    }
    (out / MANIFEST).write_text(json.dumps(record_manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(records)} official NNUE files, total "
          f"{sum(i['bytes'] for i in records):,} bytes", flush=True)


def verify(out: Path) -> list[dict]:
    data = entries()
    manifest_file = out / MANIFEST
    if not manifest_file.is_file():
        raise ValueError(f"Missing downloaded NNUE manifest: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != 1 or manifest.get("source_catalog") != data["source"]:
        raise ValueError("NNUE manifest source or format mismatch")
    records = manifest.get("files")
    if not isinstance(records, list) or len(records) != len(data["networks"]):
        raise ValueError("Missing/extra networks in NNUE manifest")
    for entry, record in zip(data["networks"], records):
        inspected = inspect(out / "networks" / entry["filename"], entry["filename"])
        expected = {**inspected, "variant": entry["variant"], "source_url": entry["url"],
                    "author": entry.get("author", "")}
        if record != expected:
            raise ValueError(f"NNUE manifest does not match verified files: {entry['filename']}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("download", "verify"))
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "fsf-networks")
    args = parser.parse_args()
    if args.command == "download":
        download(args.output)
    else:
        records = verify(args.output)
        print(f"Verified {len(records)} downloaded FSF NNUE networks", flush=True)


if __name__ == "__main__":
    main()
