from __future__ import annotations

import shutil
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from capabilities import CapabilityError, get_capability


def _check_parameters(schema: dict[str, Any], parameters: dict[str, Any]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    unknown = sorted(set(parameters).difference(schema))
    checks.append({"name": "parameters_known", "passed": not unknown, "detail": unknown})
    for name, value in parameters.items():
        rule = schema.get(name)
        if not rule:
            continue
        expected_type = rule.get("type")
        passed = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
        if expected_type == "integer":
            passed = passed and isinstance(value, int)
        if not passed:
            checks.append({"name": f"parameter:{name}", "passed": False, "detail": value})
            continue
        if "minimum" in rule and value < rule["minimum"]:
            passed = False
        if "maximum" in rule and value > rule["maximum"]:
            passed = False
        if "enum" in rule and value not in rule["enum"]:
            passed = False
        checks.append({"name": f"parameter:{name}", "passed": passed, "detail": value})
    return checks


def _gpu_check() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"name": "gpu_runtime", "passed": False, "detail": "nvidia-smi not found"}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"name": "gpu_runtime", "passed": False, "detail": str(exc)}
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return {"name": "gpu_runtime", "passed": bool(devices), "detail": devices}


def run_preflight(
    capability_id: str,
    model_root: Path,
    data_type: str,
    method: str,
    parameters: dict[str, Any],
    *,
    check_gpu: bool = True,
) -> dict[str, Any]:
    try:
        capability = get_capability(capability_id)
    except CapabilityError as exc:
        return {"passed": False, "checks": [{"name": "capability", "passed": False, "detail": str(exc)}]}

    model_path = model_root / capability["model_name"]
    checks = [
        {
            "name": "validation_status",
            "passed": capability["validation_status"] in {"gpu_smoke_passed", "legacy"},
            "detail": capability["validation_status"],
        },
        {"name": "data_type", "passed": data_type in capability["input_types"], "detail": data_type},
        {"name": "training_method", "passed": method in capability["training_methods"], "detail": method},
        {"name": "model_path", "passed": model_weights_ready(model_path), "detail": str(model_path)},
        {"name": "trainer", "passed": shutil.which("llamafactory-cli") is not None, "detail": "llamafactory-cli"},
    ]
    checks.extend(_check_parameters(capability["parameter_schema"], parameters))
    if check_gpu:
        checks.append(_gpu_check())
    return {
        "passed": all(item["passed"] for item in checks),
        "capability": capability,
        "model_path": str(model_path),
        "checks": checks,
    }


def model_weights_ready(model_path: Path) -> bool:
    try:
        if not (model_path / "config.json").is_file():
            return False
        index = model_path / "model.safetensors.index.json"
        weights = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else {"model.safetensors"}
        return bool(weights) and all((model_path / name).is_file() and (model_path / name).stat().st_size > 0 and (model_path / name).resolve().is_relative_to(model_path.resolve()) for name in weights)
    except (OSError, ValueError, KeyError, TypeError):
        return False
