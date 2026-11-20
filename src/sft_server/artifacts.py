from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_artifact_report(output_dir: Path, data_type: str, method: str) -> dict[str, Any]:
    required = ["adapter_config.json", "adapter_model.safetensors"] if method == "lora" else ["config.json"]
    if data_type in {"image_text", "video_text"}:
        processor_groups = [
            ("processor_config.json", "preprocessor_config.json"),
            ("tokenizer_config.json",),
        ]
    else:
        processor_groups = [("tokenizer_config.json",)]
    missing = [name for name in required if not (output_dir / name).is_file() or not (output_dir / name).stat().st_size]
    if method != "lora" and not any((output_dir / name).is_file() for name in ("model.safetensors", "model.safetensors.index.json")):
        # Freeze and full training save the whole model, in one file or sharded with an index.
        missing.append("model.safetensors|model.safetensors.index.json")
    for alternatives in processor_groups:
        if not any((output_dir / name).is_file() for name in alternatives):
            missing.append("|".join(alternatives))
    files = []
    if output_dir.is_dir():
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.stat().st_size > 0:
                files.append({"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)})
    return {"complete": not missing, "missing": missing, "files": files}
