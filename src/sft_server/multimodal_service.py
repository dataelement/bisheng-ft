"""Single GPU service implementing Bisheng's FT contract with persistent real jobs."""
from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import threading

import requests
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from fastapi.responses import JSONResponse

from capabilities import CapabilityError, get_capability, public_capabilities
from dataset_package import DatasetPackageError
from preflight import run_preflight, model_weights_ready
from v2_runtime import V2RuntimeError, run_inference, run_v2_job

ROOT = Path(os.getenv('BISHENG_FT_OUTPUT_ROOT', '/data/jobs'))
MODELS = Path(os.getenv('BISHENG_FT_MODEL_ROOT', '/models'))
LOCK = threading.RLock()
active_job = None
inference_busy = False
processes = {}
ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$')


def directory(identifier):
    if not isinstance(identifier, str) or not ID.fullmatch(identifier):
        raise HTTPException(422, 'invalid job_id')
    return ROOT / identifier


def state(identifier):
    target = directory(identifier) / 'service-state.json'
    if not target.is_file():
        raise HTTPException(404, 'job not found')
    return json.loads(target.read_text())


def save(identifier, value):
    target = directory(identifier) / 'service-state.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(target)


def response(data=None):
    return {'status_code': 200, 'status_message': 'success', 'data': data}


@asynccontextmanager
async def lifespan(_app):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / '.service.lock').open('w') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path in ROOT.glob('*/service-state.json'):
            value = json.loads(path.read_text())
            if value['status'] == 'RUNNING':
                value.update(status='FAILED', stage='interrupted', reason='service restarted before confirmed completion; inspect logs before retry')
                save(path.parent.name, value)
        yield
        with LOCK:
            for process in processes.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)


app = FastAPI(title='Bisheng FT text, image and video service', version='2.1.0', lifespan=lifespan)


@app.exception_handler(ValidationError)
def invalid_payload(_request, exc):
    return JSONResponse(status_code=422, content={'detail': exc.errors(include_context=False)})


class Preflight(BaseModel):
    model_config = ConfigDict(extra='forbid')
    capability_id: str
    data_type: Literal['text', 'image_text', 'video_text']
    method: Literal['lora']
    parameters: dict = Field(default_factory=dict)


class Create(BaseModel):
    model_config = ConfigDict(extra='forbid')
    job_id: str
    options: list[str] = Field(default_factory=list)
    params: dict


class Generation(BaseModel):
    model_config = ConfigDict(extra='forbid')
    max_new_tokens: int = Field(default=160, ge=1, le=512)


class Inference(BaseModel):
    model_config = ConfigDict(extra='forbid')
    prompt: str = Field(min_length=1, max_length=8000)
    media_url: str | None = None
    baseline: bool = False
    generation: Generation = Field(default_factory=Generation)


@app.get('/health')
def health():
    return response({'contract_version': 2, 'active_job_id': active_job, 'inference_busy': inference_busy})


@app.get('/v2.1/sft/capabilities')
def capabilities():
    catalog = public_capabilities()
    for item in catalog['models']:
        item['ready'] = model_weights_ready(MODELS / item['model_name'])
    return response(catalog)


@app.get('/v2.1/sft/model')
def models():
    return response(sorted({m['model_name'] for m in capabilities()['data']['models'] if m['ready']}))


@app.get('/v2.1/sft/gpu')
def gpu():
    result = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True, timeout=10)
    if result.returncode:
        raise HTTPException(503, result.stderr[-2000:])
    return response(result.stdout)


@app.post('/v2.1/sft/preflight')
def preflight(request: Preflight):
    result = run_preflight(request.capability_id, MODELS, request.data_type, request.method, request.parameters)
    with LOCK:
        result['checks'].append({'name': 'service_available', 'passed': not active_job and not inference_busy, 'detail': active_job})
    result['passed'] = all(c['passed'] for c in result['checks'])
    return response(result)


def worker(identifier, parameters):
    global active_job
    def cancelled():
        with LOCK:
            return state(identifier).get('cancel_requested', False)
    def started(process):
        with LOCK:
            processes[identifier] = process
            value = state(identifier)
            value.update(stage='training', pid=process.pid)
            save(identifier, value)
            if value.get('cancel_requested'):
                os.killpg(process.pid, signal.SIGTERM)
    try:
        code, detail = run_v2_job(identifier, parameters, directory(identifier), MODELS, started, cancelled)
        with LOCK:
            value = state(identifier)
            value.update(detail)
            value['status'] = 'CANCELLED' if value.get('cancel_requested') else ('FINISHED' if code == 0 else 'FAILED')
            if code != 0:
                stderr = directory(identifier) / 'job_result' / 'stderr'
                value['reason'] = stderr.read_text(errors='replace')[-4000:] if stderr.is_file() else 'training failed'
            save(identifier, value)
    except Exception as exc:
        with LOCK:
            value = state(identifier)
            value.update(status='CANCELLED' if value.get('cancel_requested') else 'FAILED', stage='failed', reason=str(exc))
            save(identifier, value)
    finally:
        with LOCK:
            value = state(identifier)
            value['finished_at'] = datetime.now(timezone.utc).isoformat()
            save(identifier, value)
            processes.pop(identifier, None)
            active_job = None


@app.post('/v2.1/sft/job')
def create(request: Create):
    global active_job
    directory(request.job_id)
    params = request.params
    if request.options not in ([], ['train']) or params.get('contract_version') != 2:
        raise HTTPException(422, 'this service requires contract_version=2 and train options')
    try:
        capability = get_capability(params.get('capability_id'))
    except CapabilityError as exc:
        raise HTTPException(422, str(exc)) from exc
    metadata = {'contract_version', 'capability_id', 'data_type', 'finetuning_type', 'model_name_or_path', 'dataset'}
    parameters = {k: v for k, v in params.items() if k not in metadata}
    if params.get('model_name_or_path') != capability['model_name']:
        raise HTTPException(422, 'model does not match capability')
    if not isinstance(params.get('dataset'), str) or not params['dataset']:
        raise HTTPException(422, 'dataset URL is required')
    checks = run_preflight(capability['capability_id'], MODELS, params.get('data_type'), params.get('finetuning_type'), parameters)
    if not checks['passed']:
        raise HTTPException(422, checks)
    with LOCK:
        if active_job or inference_busy:
            raise HTTPException(409, 'training or inference is already running')
        if directory(request.job_id).exists():
            raise HTTPException(409, 'job_id already exists; query original job')
        active_job = request.job_id
        save(request.job_id, {'job_id': request.job_id, 'status': 'RUNNING', 'stage': 'preparing', 'params': params,
                              'created_at': datetime.now(timezone.utc).isoformat(), 'reason': ''})
        threading.Thread(target=worker, args=(request.job_id, params), daemon=True).start()
    return response({'job_id': request.job_id})


@app.post('/v2.1/sft/job/status')
def status(payload: dict):
    with LOCK:
        return response(state(payload.get('job_id')))


@app.post('/v2.1/sft/job/cancel')
def cancel(payload: dict):
    identifier = payload.get('job_id')
    with LOCK:
        value = state(identifier)
        if value['status'] != 'RUNNING':
            return response(value)
        value['cancel_requested'] = True
        value['stage'] = 'cancelling'
        save(identifier, value)
        process = processes.get(identifier)
        if process and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    return response({'status': 'CANCELLING'})


@app.post('/v2.1/sft/job/log')
def logs(payload: dict):
    identifier = payload.get('job_id')
    state(identifier)
    folder = directory(identifier)
    def read(path):
        return path.read_text(errors='replace')[-1000000:] if path.is_file() else ''
    return response({'log_data': read(folder / 'model_output' / 'trainer_log.jsonl'),
                     'train_log': read(folder / 'job_result' / 'stdout') + read(folder / 'job_result' / 'stderr')})


@app.post('/v2.1/sft/job/metrics')
def metrics(payload: dict):
    identifier = payload.get('job_id')
    state(identifier)
    report = directory(identifier) / 'model_output' / 'all_results.json'
    return response({'report': json.loads(report.read_text()) if report.is_file() else {}})


@app.post('/v2.1/sft/job/{job_id}/infer')
def infer(job_id: str, request: Inference):
    global inference_busy
    with LOCK:
        value = state(job_id)
        if value['status'] != 'FINISHED':
            raise HTTPException(409, 'training has not finished successfully')
        if active_job or inference_busy:
            raise HTTPException(409, 'training or inference is already running')
        if value['params']['data_type'] != 'text' and not request.media_url:
            raise HTTPException(422, 'media_url is required')
        inference_busy = True
    try:
        return response(run_inference(directory(job_id), request.model_dump()))
    except (V2RuntimeError, DatasetPackageError, requests.RequestException) as exc:
        raise HTTPException(422, str(exc)) from exc
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(504, "inference exceeded the 300 second timeout; inspect saved process logs") from exc
    finally:
        with LOCK:
            inference_busy = False


@app.post('/v2.1/sft/job/delete')
def delete(payload: dict):
    identifier = payload.get('job_id')
    with LOCK:
        value = state(identifier)
        if value['status'] == 'RUNNING' or inference_busy:
            raise HTTPException(409, 'wait for cancellation or inference completion')
        shutil.rmtree(directory(identifier))
    return response()


@app.post('/v2.1/models/sft_elem/infer')
def ingress(payload: dict):
    uri = payload.get('uri')
    if uri == '/v2.1/sft/job':
        return create(Create.model_validate({k: v for k, v in payload.items() if k != 'uri'}))
    if uri == '/v2.1/sft/preflight':
        return preflight(Preflight.model_validate(payload.get('params', {})))
    if uri == '/v2.1/sft/capabilities':
        return capabilities()
    actions = {'status': status, 'cancel': cancel, 'log': logs, 'metrics': metrics, 'delete': delete}
    for name, action in actions.items():
        if uri == f'/v2.1/sft/job/{name}':
            return action(payload)
    raise HTTPException(422, 'unsupported operation; adapter publication requires a separate serving deployment')
