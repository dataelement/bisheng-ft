from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
import zipfile
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

import requests
import yaml

import remote_source
import uploads
from artifacts import build_artifact_report
from capabilities import get_capability
from dataset_package import (
    _read_records,
    _safe_member,
    dataset_byte_limit,
    dataset_file_limit,
    validate_media,
    validate_multimodal_dir,
    validate_multimodal_zip,
    validate_text_json,
)


class V2RuntimeError(RuntimeError):
    pass


def _download(url: str, destination: Path) -> None:
    try:
        pushed = uploads.resolve(url)
    except uploads.UploadError as exc:
        raise V2RuntimeError(str(exc)) from exc
    if pushed is not None:
        shutil.copyfile(pushed, destination)
        return
    if urlparse(url).scheme not in {"http", "https"}:
        raise V2RuntimeError("dataset/media URL must use HTTP or HTTPS")
    limit = uploads.size_limit()
    received = 0
    with requests.get(url, stream=True, timeout=(10, 300)) as response:
        response.raise_for_status()
        with destination.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                if block:
                    received += len(block)
                    if received > limit:
                        raise V2RuntimeError("download exceeds configured size limit")
                    stream.write(block)


def _unpack(source: Path, data_type: str, train_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Turn one downloaded package (ZIP, JSON or JSONL) into a dataset directory with train.json."""
    data_dir = train_dir / "dataset"
    if data_type in {"image_text", "video_text"}:
        return data_dir, validate_multimodal_zip(source, data_dir, data_type)
    data_dir.mkdir()
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            members = archive.infolist()
            if len(members) > dataset_file_limit() or sum(item.file_size for item in members) > dataset_byte_limit():
                raise V2RuntimeError("text package exceeds size limit")
            if len({i.filename for i in members}) != len(members) or any(not _safe_member(i.filename) or (i.external_attr >> 16) & 0o170000 == 0o120000 for i in members):
                raise V2RuntimeError("unsafe or duplicate archive member")
            manifests = [name for name in ('train.json', 'train.jsonl', 'data.json', 'data.jsonl') if name in archive.namelist()]
            if len(manifests) != 1:
                raise V2RuntimeError("text package must contain exactly one root manifest")
            raw = data_dir / "source.json"
            raw.write_bytes(archive.read(manifests[0]))
    else:
        raw = source
    report = validate_text_json(raw)
    (data_dir / "train.json").write_text(json.dumps(_read_records(raw), ensure_ascii=False))
    return data_dir, report


def _prepare_remote(url: str, data_type: str, train_dir: Path, source: dict, cancelled=None) -> tuple[Path, dict[str, Any]]:
    """Fetch the dataset straight from the customer's storage using the plan recorded at import."""
    plan_path = train_dir / "plan.json"
    _download(url, plan_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("kind") != "remote_plan" or plan.get("data_type") != data_type:
        raise V2RuntimeError("remote dataset plan does not match this job")
    limit = dataset_byte_limit()
    try:
        if plan["layout"] == "zip":
            archive = plan["zip"]
            fetched = remote_source.fetch(source, [{"remote": archive["path"], "local": "remote.zip", "size": archive["size"]}], train_dir, limit, cancelled)
            data_dir, report = _unpack(train_dir / "remote.zip", data_type, train_dir)
        else:
            data_dir = train_dir / "dataset"
            data_dir.mkdir()
            (data_dir / "train.json").write_text(json.dumps(plan["records"], ensure_ascii=False), encoding="utf-8")
            files = [{"remote": remote_source.remote_path(source, item["path"]), "local": item["path"], "size": item["size"]} for item in plan.get("files", [])]
            if any(not _safe_member(item["local"]) for item in files):
                raise V2RuntimeError("remote dataset plan has an unsafe media path")
            if len(files) > dataset_file_limit():
                raise V2RuntimeError("remote dataset has more files than this training host allows")
            fetched = remote_source.fetch(source, files, data_dir, limit, cancelled)
            report = validate_text_json(data_dir / "train.json") if data_type == "text" else validate_multimodal_dir(data_dir, data_type)
    except remote_source.RemoteSourceError as exc:
        raise V2RuntimeError(str(exc)) from exc
    report.update(source=source["type"], fetched_bytes=fetched)
    return data_dir, report


def _write_dataset_info(data_dir: Path, data_type: str) -> None:
    if data_type in {"image_text", "video_text"}:
        media_field = "images" if data_type == "image_text" else "videos"
        info = {"file_name": "train.json", "formatting": "sharegpt", "columns": {"messages": "messages", media_field: media_field}}
    else:
        records = _read_records(data_dir / "train.json")
        info = {"file_name": "train.json"}
        if records and "messages" in records[0]:
            info.update({"formatting": "sharegpt", "columns": {"messages": "messages"}})
    if info.get("formatting") == "sharegpt":
        info["tags"] = {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system",
        }
    (data_dir / "dataset_info.json").write_text(
        json.dumps({"bisheng_train": info}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_dataset(urls: list[str], data_type: str, train_dir: Path, source: dict | None = None, cancelled=None) -> tuple[Path, dict[str, Any]]:
    if len(urls) != 1:
        raise V2RuntimeError("training requires exactly one dataset package")
    if source is not None:
        data_dir, report = _prepare_remote(urls[0], data_type, train_dir, source, cancelled)
    else:
        package = train_dir / ("dataset.zip" if data_type in {"image_text", "video_text"} else "train.json")
        _download(urls[0], package)
        data_dir, report = _unpack(package, data_type, train_dir)
    _write_dataset_info(data_dir, data_type)
    return data_dir, report


def build_train_config(params: dict[str, Any], model_path: Path, data_dir: Path, output_dir: Path) -> dict[str, Any]:
    capability = get_capability(params["capability_id"])
    if capability["validation_status"] != "gpu_smoke_passed":
        raise V2RuntimeError("capability has not passed real GPU validation")
    data_type = params["data_type"]
    if data_type not in capability["input_types"] or params.get("finetuning_type") != "lora":
        raise V2RuntimeError("requested modality or training method is not supported")
    config = {
        "stage": "sft",
        "do_train": True,
        "model_name_or_path": str(model_path),
        "dataset": "bisheng_train",
        "dataset_dir": str(data_dir),
        "template": capability["template"],
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": int(params.get("lora_rank", 8)),
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "cutoff_len": int(params.get("cutoff_len", 1024)),
        "per_device_train_batch_size": int(params.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(params.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": float(params.get("num_train_epochs", 1)),
        "learning_rate": float(params.get("learning_rate", 0.0001)),
        "max_steps": int(params.get("max_steps", 8)),
        "logging_steps": 1,
        "save_strategy": "no",
        "plot_loss": False,
        "bf16": True,
        "report_to": "none",
        "trust_remote_code": False,
        "flash_attn": "sdpa",
        "freeze_vision_tower": True,
        "freeze_multi_modal_projector": True,
        "preprocessing_num_workers": 1,
        "dataloader_num_workers": 0,
        "gradient_checkpointing": True,
        "image_max_pixels": 131072,
        "video_max_pixels": 65536,
        "video_fps": 2.0,
        "video_maxlen": 8,
    }

    if capability["model_name"].startswith("Qwen3-"):
        config["enable_thinking"] = False
    return config


def run_v2_job(job_id: str, params: dict[str, Any], job_dir: Path, model_root: Path, on_process=None, cancelled=None, source: dict | None = None) -> tuple[int, dict[str, Any]]:
    train_dir = job_dir / "train_data"
    output_dir = job_dir / "model_output"
    result_dir = job_dir / "job_result"
    for directory in (train_dir, output_dir, result_dir):
        directory.mkdir(parents=True, exist_ok=True)
    urls = [item for item in str(params.get("dataset", "")).split(",") if item]
    data_type = str(params.get("data_type", ""))
    data_dir, dataset_report = prepare_dataset(urls, data_type, train_dir, source, cancelled)
    model_path = model_root / str(params["model_name_or_path"])
    if not model_path.is_dir():
        raise V2RuntimeError(f"model path does not exist: {model_path}")
    if cancelled and cancelled():
        raise V2RuntimeError("job cancelled before training")
    config = build_train_config(params, model_path, data_dir, output_dir)
    config_path = result_dir / "train.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    stdout_path = result_dir / "stdout"
    stderr_path = result_dir / "stderr"
    pid_path = result_dir / "pid"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(["llamafactory-cli", "train", str(config_path)], stdout=stdout, stderr=stderr, start_new_session=True)
        pid_path.write_text(str(process.pid), encoding="utf-8")
        if on_process:
            on_process(process)
        exit_code = process.wait()
    for name in ("tokenizer_config.json", "preprocessor_config.json", "processor_config.json"):
        if (model_path / name).is_file() and not (output_dir / name).is_file():
            shutil.copy2(model_path / name, output_dir / name)
    artifacts = build_artifact_report(output_dir, data_type, "lora")
    state = {
        "job_id": job_id,
        "exit_code": exit_code,
        "stage": "completed" if exit_code == 0 and artifacts["complete"] else "failed",
        "dataset_report": dataset_report,
        "artifacts": artifacts,
        "model_path": str(model_path),
        "data_type": data_type,
        "capability_id": params["capability_id"],
        "config": config,
    }
    (result_dir / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return exit_code if artifacts["complete"] else 1, state


def run_inference(job_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    state_path = job_dir / "job_result" / "state.json"
    if not state_path.is_file():
        raise V2RuntimeError("training state is missing")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not state.get("artifacts", {}).get("complete"):
        raise V2RuntimeError("adapter artifacts are incomplete")
    request_dir = job_dir / "job_result" / "inference" / uuid.uuid4().hex
    request_dir.mkdir(parents=True)
    request_path = request_dir / "request.json"
    response_path = request_dir / "response.json"
    payload = dict(payload)
    if state["data_type"] != "text":
        media_path = request_dir / ("sample.png" if state["data_type"] == "image_text" else "sample.mp4")
        _download(payload.get("media_url", ""), media_path)
        validate_media(media_path, state["data_type"])
        payload["media_path"] = str(media_path)
    payload["model_path"] = state["model_path"]
    payload["data_type"] = state["data_type"]
    payload["template"] = state.get("config", {}).get("template")
    if not payload["template"]:
        payload["template"] = get_capability(state["capability_id"])["template"] if state.get("capability_id") else "qwen"
    payload["adapter"] = None if payload.get("baseline") else str(job_dir / "model_output")
    request_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    command = [sys.executable, str(Path(__file__).with_name("inference_runner.py")), "--request", str(request_path), "--response", str(response_path)]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=300, env=os.environ.copy())
    (request_dir / "process.log").write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise V2RuntimeError(completed.stderr[-4000:] or "inference process failed")
    return json.loads(response_path.read_text(encoding="utf-8"))
