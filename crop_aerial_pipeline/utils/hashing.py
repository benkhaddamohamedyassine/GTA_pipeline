"""File and configuration fingerprinting, used by the resume/caching logic in
``stages/`` to decide whether a cached stage output is still valid.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def file_fingerprint(path: Path) -> Dict[str, Any]:
    """Cheap, resume-safe fingerprint of a source file: size + mtime (fast) plus
    a content hash (sha256) for a reliable "did this actually change" check.
    Images are small enough (a few MB) that hashing the full content is cheap
    relative to any of the model-inference stages that follow.
    """
    path = Path(path)
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": hash_file_content(path),
    }


def hash_file_content(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fingerprints_match(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if a is None or b is None:
        return False
    # size+mtime is the fast path; only fall back to comparing the content hash
    # if either is missing (e.g. an older manifest written before this field
    # existed) or if size/mtime already disagree (in which case sha256 will too,
    # but comparing it explicitly keeps the check meaningful even if a clock or
    # filesystem quirk makes mtime unreliable on the user's Drive mount).
    return a.get("sha256") == b.get("sha256") and a.get("size") == b.get("size")


def hash_config(config: Any) -> str:
    """Stable hash of a (dataclass) config object's field values, used to
    invalidate cached stage outputs when the user changes a relevant setting.
    """
    if dataclasses.is_dataclass(config):
        payload = dataclasses.asdict(config)
    elif isinstance(config, dict):
        payload = config
    else:
        raise TypeError(f"hash_config expects a dataclass instance or dict, got {type(config)!r}")
    normalized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
