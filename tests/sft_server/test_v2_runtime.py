import json
from pathlib import Path
from types import SimpleNamespace

import v2_runtime


def test_build_train_config_uses_validated_capability(tmp_path):
    config = v2_runtime.build_train_config(
        {
            "capability_id": "qwen2.5-1.5b-instruct-text-lora-20261030",
            "data_type": "text",
            "finetuning_type": "lora",
            "max_steps": 3,
            "lora_rank": 8,
            "learning_rate": 0.00005,
        },
        tmp_path / "model",
        tmp_path / "data",
        tmp_path / "output",
    )
    assert config["template"] == "qwen"
    assert config["max_steps"] == 3
    assert config["lora_rank"] == 8


def test_run_v2_job_requires_artifacts(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    (model_root / "Qwen2.5-1.5B-Instruct").mkdir(parents=True)
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps([{"instruction": "say hi", "input": "", "output": "hi"}]),
        encoding="utf-8",
    )

    def fake_download(_url: str, destination: Path):
        destination.write_bytes(source.read_bytes())

    class FakeProcess:
        pid = 123

        def wait(self):
            return 0

    monkeypatch.setattr(v2_runtime, "_download", fake_download)
    monkeypatch.setattr(v2_runtime.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        v2_runtime,
        "build_artifact_report",
        lambda *args: {"complete": False, "missing": ["adapter_model.safetensors"], "files": []},
    )
    code, state = v2_runtime.run_v2_job(
        "job-1",
        {
            "contract_version": 2,
            "capability_id": "qwen2.5-1.5b-instruct-text-lora-20261030",
            "data_type": "text",
            "finetuning_type": "lora",
            "model_name_or_path": "Qwen2.5-1.5B-Instruct",
            "dataset": "https://example.invalid/train.json",
        },
        tmp_path / "job",
        model_root,
    )
    assert code == 1
    assert state["stage"] == "failed"


def test_run_inference_propagates_real_process_failure(monkeypatch, tmp_path):
    job_dir = tmp_path / "job"
    result_dir = job_dir / "job_result"
    result_dir.mkdir(parents=True)
    (result_dir / "state.json").write_text(
        json.dumps(
            {
                "model_path": "/models/text",
                "data_type": "text",
                "artifacts": {"complete": True},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        v2_runtime.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stderr="CUDA out of memory", stdout=""),
    )
    try:
        v2_runtime.run_inference(job_dir, {"prompt": "test"})
    except v2_runtime.V2RuntimeError as exc:
        assert "CUDA out of memory" in str(exc)
    else:
        raise AssertionError("inference failure must not be reported as success")


def test_text_zip_jsonl_is_normalized_without_losing_samples(tmp_path, monkeypatch):
    import zipfile
    package = tmp_path / 'input.zip'
    rows = [{'messages': [{'role':'user','content':'a'}, {'role':'assistant','content':'b'}]},
            {'messages': [{'role':'user','content':'c'}, {'role':'assistant','content':'d'}]}]
    with zipfile.ZipFile(package, 'w') as archive:
        archive.writestr('data.jsonl', '\n'.join(json.dumps(r) for r in rows))
    monkeypatch.setattr(v2_runtime, '_download', lambda _url,destination: destination.write_bytes(package.read_bytes()))
    root = tmp_path / 'job'
    root.mkdir()
    directory, report = v2_runtime.prepare_dataset(['https://example.invalid/data.zip'], 'text', root)
    assert report['samples'] == 2
    assert json.loads((directory/'train.json').read_text()) == rows
    assert json.loads((directory/'dataset_info.json').read_text())['bisheng_train']['formatting'] == 'sharegpt'


def test_epoch_training_does_not_force_positive_step_limit(tmp_path):
    config = v2_runtime.build_train_config({'capability_id':'qwen3-0.6b-lora-20261102', 'data_type':'text',
            'finetuning_type':'lora','max_steps':-1,'num_train_epochs':3},tmp_path/'model',tmp_path/'data',tmp_path/'output')
    assert config['max_steps'] == -1
    assert config['num_train_epochs'] == 3
    assert config['enable_thinking'] is False


def test_inference_prompt_keeps_a_single_media_placeholder():
    from inference_runner import build_prompt
    assert build_prompt('What color?', 'image') == '<image>What color?'
    assert build_prompt('<image>What color?', 'image') == '<image>What color?'
    assert build_prompt('<video>Which way?', 'video') == '<video>Which way?'
    assert build_prompt('Classify this', None) == 'Classify this'
