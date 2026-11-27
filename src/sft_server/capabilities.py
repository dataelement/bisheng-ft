from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


CAPABILITY_FILE = Path(__file__).with_name("model_capabilities.json")
ALLOWED_STATUSES = {"gpu_smoke_passed", "legacy", "unverified", "blocked"}


class CapabilityError(ValueError):
    pass


def load_capability_catalog(path: Path = CAPABILITY_FILE) -> dict[str, Any]:
    path = Path(os.getenv("BISHENG_FT_CAPABILITIES", str(path)))
    catalog = json.loads(path.read_text(encoding="utf-8"))
    if catalog.get("contract_version") != 2:
        raise CapabilityError("unsupported capability contract version")
    seen: set[str] = set()
    for item in catalog.get("models", []):
        required = {
            "capability_id",
            "model_name",
            "model_source",
            "input_types",
            "training_methods",
            "template",
            "validation_status",
            "parameter_schema",
        }
        missing = required.difference(item)
        if missing:
            raise CapabilityError(f"capability is missing fields: {sorted(missing)}")
        identifier = item["capability_id"]
        if identifier in seen:
            raise CapabilityError(f"duplicate capability_id: {identifier}")
        if item["validation_status"] not in ALLOWED_STATUSES:
            raise CapabilityError(f"invalid validation_status for {identifier}")
        seen.add(identifier)
    allowed = [name for name in os.getenv("BISHENG_FT_MODELS", "").split(",") if name]
    if allowed:
        catalog["models"] = [m for m in catalog["models"] if m["model_name"] in allowed]
    return catalog


def get_capability(capability_id: str, path: Path = CAPABILITY_FILE) -> dict[str, Any]:
    for item in load_capability_catalog(path)["models"]:
        if item["capability_id"] == capability_id:
            return item
    raise CapabilityError(f"unknown capability_id: {capability_id}")


def public_capabilities(path: Path = CAPABILITY_FILE) -> dict[str, Any]:
    catalog = load_capability_catalog(path)
    return {
        "contract_version": catalog["contract_version"],
        "models": [{key: value for key, value in item.items() if key != "model_path"} for item in catalog["models"]],
    }
