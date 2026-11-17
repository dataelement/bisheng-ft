from artifacts import build_artifact_report


def test_text_lora_artifacts_are_complete(tmp_path):
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    report = build_artifact_report(tmp_path, "text", "lora")
    assert report["complete"] is True
    assert len(report["files"]) == 3


def test_image_lora_requires_processor(tmp_path):
    for name in ("adapter_config.json", "adapter_model.safetensors", "tokenizer_config.json"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    report = build_artifact_report(tmp_path, "image_text", "lora")
    assert report["complete"] is False
    assert "processor_config.json|preprocessor_config.json" in report["missing"]
