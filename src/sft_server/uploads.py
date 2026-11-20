"""Content-addressed files pushed by the platform when this host cannot download its storage links."""
from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path

SCHEME = "ft-file://"
DIGEST = re.compile(r"^[0-9a-f]{64}$")
ROOT = Path(os.getenv("BISHENG_FT_OUTPUT_ROOT", "/data/jobs")) / "_uploads"


class UploadError(ValueError):
    pass


def configure(root: Path) -> None:
    global ROOT
    ROOT = root


def size_limit() -> int:
    # One stored package (pushed or pulled); the platform accepts local uploads up to 2 GiB.
    return int(os.getenv("BISHENG_FT_MAX_PACKAGE_BYTES", str(2 * 1024 ** 3 + 16 * 1024 ** 2)))


def path_for(digest: str) -> Path:
    if not isinstance(digest, str) or not DIGEST.fullmatch(digest):
        raise UploadError("file reference must be a lowercase sha256 digest")
    return ROOT / digest


async def store(digest: str, chunks) -> dict:
    """Write the streamed body under its digest; an existing file is reused so retries cost nothing."""
    target = path_for(digest)
    if target.is_file():
        return {"sha256": digest, "size": target.stat().st_size, "stored": False}
    ROOT.mkdir(parents=True, exist_ok=True)
    temporary = ROOT / f".{digest}.{uuid.uuid4().hex}.part"
    hasher = hashlib.sha256()
    size = 0
    try:
        with temporary.open("wb") as stream:
            async for chunk in chunks:
                size += len(chunk)
                if size > size_limit():
                    raise UploadError("upload exceeds configured size limit")
                hasher.update(chunk)
                stream.write(chunk)
        if size == 0:
            raise UploadError("upload is empty")
        if hasher.hexdigest() != digest:
            raise UploadError("sha256 does not match the uploaded content")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return {"sha256": digest, "size": size, "stored": True}


def resolve(reference: str) -> Path | None:
    """Return the local file for an ft-file:// reference, or None for an ordinary URL."""
    if not reference.startswith(SCHEME):
        return None
    target = path_for(reference[len(SCHEME):])
    if not target.is_file():
        raise UploadError("pushed file is missing on the FT host; submit the task again to re-upload it")
    return target
