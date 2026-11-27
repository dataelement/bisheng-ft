import preflight


TEXT_ID = "qwen2.5-1.5b-instruct-text-lora-20261030"
VIDEO_ID = "qwen2.5-vl-3b-video-lora-unverified"


def test_preflight_passes_exact_validated_combination(tmp_path, monkeypatch):
    (tmp_path / "Qwen2.5-1.5B-Instruct").mkdir()
    monkeypatch.setattr(preflight, "model_weights_ready", lambda path: True)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/bin/{name}")
    result = preflight.run_preflight(TEXT_ID, tmp_path, "text", "lora", {"max_steps": 4}, check_gpu=False)
    assert result["passed"] is True


def test_preflight_blocks_wrong_modality_and_unknown_parameter(tmp_path, monkeypatch):
    (tmp_path / "Qwen2.5-1.5B-Instruct").mkdir()
    monkeypatch.setattr(preflight, "model_weights_ready", lambda path: True)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/bin/{name}")
    result = preflight.run_preflight(TEXT_ID, tmp_path, "image_text", "lora", {"shell": "bad"}, check_gpu=False)
    assert result["passed"] is False
    assert {item["name"] for item in result["checks"] if not item["passed"]} == {"data_type", "parameters_known"}


def test_preflight_blocks_unverified_model(tmp_path, monkeypatch):
    (tmp_path / "Qwen2.5-VL-3B-Instruct").mkdir()
    from capabilities import get_capability
    pending = dict(get_capability("qwen2.5-vl-3b-instruct-lora-20261102"), validation_status="unverified")
    monkeypatch.setattr(preflight, "get_capability", lambda identifier: pending)
    monkeypatch.setattr(preflight, "model_weights_ready", lambda path: True)
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/bin/{name}")
    result = preflight.run_preflight(VIDEO_ID, tmp_path, "video_text", "lora", {}, check_gpu=False)
    assert result["passed"] is False
    assert next(item for item in result["checks"] if item["name"] == "validation_status")["detail"] == "unverified"
