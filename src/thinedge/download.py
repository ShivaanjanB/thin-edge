"""Download Cricsheet ball-by-ball JSON archives into data/raw/<source>/.

Design notes
- Downloads are cached: a source already in the manifest is skipped unless ``refresh=True``.
  Cricsheet is a volunteer-run site, so we fetch each archive once and politely.
- Archives are untrusted input. Only flat ``*.json`` members with safe names are extracted,
  and member count and size are capped, so a malformed archive cannot write outside its folder.
- Each source gets its own directory, and the manifest records URL, hash and time.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from .config import CRICSHEET_BASE, RAW_DIR

_SOURCE_RE = re.compile(r"^[a-z0-9_]+$")
_MEMBER_RE = re.compile(r"^[A-Za-z0-9_\-]+\.json$")

MAX_ZIP_BYTES = 500 * 1024 * 1024
MAX_MEMBER_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024  # the largest archive here is about 100 MB unpacked
MAX_MEMBERS = 50_000
USER_AGENT = "thinedge-research/0.1 (cricket data research; cached single download)"


def source_url(name: str) -> str:
    """URL of the Cricsheet JSON archive for a source such as ``ipl`` or ``hnd``."""
    if not _SOURCE_RE.fullmatch(name):
        raise ValueError(f"invalid source name: {name!r}")
    return f"{CRICSHEET_BASE}{name}_json.zip"


def _manifest_path(raw_dir: Path) -> Path:
    return raw_dir / "manifest.json"


def read_manifest(raw_dir: Path = RAW_DIR) -> dict:
    path = _manifest_path(raw_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(raw_dir: Path, manifest: dict) -> None:
    tmp = _manifest_path(raw_dir).with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, _manifest_path(raw_dir))


def extract_json_members(zip_path: Path, dest: Path) -> int:
    """Extract flat, safely named ``.json`` members of ``zip_path`` into ``dest``."""
    written = 0
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ValueError(f"archive has {len(infos)} members (limit {MAX_MEMBERS})")
        wanted = [i for i in infos if not i.is_dir() and _MEMBER_RE.fullmatch(i.filename)]
        # README.txt, nested paths and odd names are ignored. Of the rest:
        total = sum(i.file_size for i in wanted)
        if total > MAX_TOTAL_BYTES:  # many small-looking members can still add up (a zip bomb)
            raise ValueError(f"archive would extract to {total} bytes (limit {MAX_TOTAL_BYTES})")
        lower = [i.filename.lower() for i in wanted]
        if len(set(lower)) != len(lower):  # on Windows and macOS the second would overwrite the first
            raise ValueError("archive has member names that differ only in letter case")
        for info in wanted:
            if info.file_size > MAX_MEMBER_BYTES:
                raise ValueError(f"{info.filename} is {info.file_size} bytes (limit {MAX_MEMBER_BYTES})")
            (dest / info.filename).write_bytes(zf.read(info))
            written += 1
    return written


def download_source(name: str, raw_dir: Path = RAW_DIR, refresh: bool = False) -> dict:
    """Fetch one archive and extract it. Returns its manifest entry."""
    url = source_url(name)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(raw_dir)
    target = raw_dir / name
    if name in manifest and target.is_dir() and not refresh:
        return {**manifest[name], "skipped": True}

    sha = hashlib.sha256()
    size = 0
    with tempfile.TemporaryDirectory(dir=raw_dir) as tmp:
        zip_path = Path(tmp) / f"{name}.zip"
        request = Request(url, headers={"User-Agent": USER_AGENT})
        with urlopen(request, timeout=120) as resp, zip_path.open("wb") as out:
            while chunk := resp.read(1 << 20):
                size += len(chunk)
                if size > MAX_ZIP_BYTES:
                    raise ValueError(f"{url} exceeds {MAX_ZIP_BYTES} bytes")
                sha.update(chunk)
                out.write(chunk)

        staging = Path(tmp) / "extracted"
        staging.mkdir()
        n_json = extract_json_members(zip_path, staging)
        if n_json == 0:
            raise ValueError(f"{url} contained no JSON match files")
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(staging), str(target))

    entry = {
        "url": url,
        "sha256": sha.hexdigest(),
        "bytes": size,
        "n_json": n_json,
        "downloaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest[name] = entry
    _write_manifest(raw_dir, manifest)
    return {**entry, "skipped": False}
