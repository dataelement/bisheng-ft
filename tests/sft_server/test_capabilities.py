import json

import pytest
from capabilities import CapabilityError, get_capability, load_capability_catalog, public_capabilities


def test_catalog_exposes_validated_text_and_image_models():
    models = public_capabilities()["models"]
    validated = {item["model_name"]: item for item in models if item["validation_status"] == "gpu_smoke_passed"}
    assert validated["Qwen2.5-1.5B-Instruct"]["input_types"] == ["text"]
    assert validated["Qwen2-VL-2B-Instruct"]["input_types"] == ["image_text"]
    assert all("model_path" not in item for item in models)


def test_unknown_capability_is_rejected():
    with pytest.raises(CapabilityError, match="unknown capability_id"):
        get_capability("qwen-by-name-guess")


def test_duplicate_identifier_is_rejected(tmp_path):
    item = get_capability("qwen2.5-1.5b-instruct-text-lora-20261030")
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps({"contract_version": 2, "models": [item, item]}), encoding="utf-8")
    with pytest.raises(CapabilityError, match="duplicate"):
        load_capability_catalog(path)
