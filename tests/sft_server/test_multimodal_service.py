import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
import multimodal_service as service
import preflight
import v2_runtime

TEXT = 'qwen3-0.6b-lora-20261102'
VISION = 'qwen2.5-vl-3b-instruct-lora-20261102'


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'ROOT', tmp_path / 'jobs')
    monkeypatch.setattr(service, 'MODELS', tmp_path / 'models')
    monkeypatch.setattr(service, 'active_job', None)
    monkeypatch.setattr(service, 'inference_busy', False)
    monkeypatch.setattr(service, 'run_preflight', lambda *args, **kwargs: {'passed': True, 'checks': []})
    with TestClient(service.app) as client:
        yield client


def payload(identifier='job-1'):
    return {'job_id': identifier, 'options': ['train'], 'params': {'contract_version': 2, 'capability_id': TEXT,
            'model_name_or_path': 'Qwen3-0.6B', 'data_type': 'text', 'finetuning_type': 'lora',
            'dataset': 'https://store.example/data.json', 'max_steps': 2}}


def test_submit_persist_error_and_duplicate(client, monkeypatch):
    def failed(*args):
        raise ValueError('real download error')
    monkeypatch.setattr(service, 'run_v2_job', failed)
    result = client.post('/v2.1/models/sft_elem/infer', json={'uri': '/v2.1/sft/job', **payload()})
    assert result.status_code == 200
    for _ in range(100):
        current = client.post('/v2.1/sft/job/status', json={'job_id': 'job-1'}).json()['data']
        if current['status'] == 'FAILED':
            break
        time.sleep(.01)
    assert current['reason'] == 'real download error'
    assert client.post('/v2.1/sft/job', json=payload()).status_code == 409


def test_cancel_during_preparation_and_release_lock(client, monkeypatch):
    ready = threading.Event()
    proceed = threading.Event()
    def run(*args):
        ready.set()
        assert proceed.wait(2)
        assert args[-1]() is True
        raise ValueError('cancelled')
    monkeypatch.setattr(service, 'run_v2_job', run)
    assert client.post('/v2.1/sft/job', json=payload()).status_code == 200
    assert ready.wait(1)
    assert client.post('/v2.1/sft/job', json=payload('job-2')).status_code == 409
    assert client.post('/v2.1/sft/job/cancel', json={'job_id': 'job-1'}).status_code == 200
    proceed.set()
    for _ in range(100):
        if service.active_job is None:
            break
        time.sleep(.01)
    assert service.state('job-1')['status'] == 'CANCELLED'
    assert service.active_job is None


def test_unknown_state_and_path_do_not_create_directory(client):
    assert client.post('/v2.1/sft/job/status', json={'job_id': 'missing'}).status_code == 404
    assert client.post('/v2.1/sft/job/status', json={'job_id': '../escape'}).status_code == 422
    assert not (service.ROOT / 'missing').exists()


def test_video_config_and_all_parameters():
    config = v2_runtime.build_train_config({'capability_id': VISION, 'data_type': 'video_text', 'finetuning_type': 'lora',
        'max_steps': 5, 'num_train_epochs': 2, 'cutoff_len': 512, 'per_device_train_batch_size': 2,
        'gradient_accumulation_steps': 4, 'lora_rank': 16, 'learning_rate': .0002}, Path('/models/a'), Path('/data'), Path('/out'))
    assert config['max_steps'] == 5 and config['num_train_epochs'] == 2
    assert config['per_device_train_batch_size'] == 2 and config['gradient_accumulation_steps'] == 4
    assert config['cutoff_len'] == 512 and 'max_samples' not in config
    assert config['freeze_vision_tower'] and config['video_maxlen'] == 8


def test_parameter_types_fail_cleanly():
    checks = preflight._check_parameters({'max_steps': {'type':'integer', 'minimum':1}}, {'max_steps':'8'})
    assert checks[-1]['passed'] is False
