import io
import json
import socket
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import multimodal_service as service
import remote_source
import v2_runtime

SECRET = 'example-private'


def png(color):
    buffer = io.BytesIO()
    Image.new('RGB', (8, 8), color).save(buffer, 'PNG')
    return buffer.getvalue()


class FakeSession:
    def __init__(self, files):
        self.files = files

    def size(self, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return len(self.files[path])

    def download(self, path, handle):
        handle.write(self.files[path])


@pytest.fixture
def storage(monkeypatch):
    files = {}
    opened = []

    @contextmanager
    def fake_open(source):
        opened.append(source)
        yield FakeSession(files)

    monkeypatch.setattr(remote_source, 'open_source', fake_open)
    return files, opened


def source(**changes):
    return {'type': 'sftp', 'endpoint': 'sftp://files.example', 'username': 'reader', 'password': SECRET,
            'media_root': 'datasets/receipts', 'probe': 'datasets/receipts/images/a.png', **changes}


def plan(tmp_path, **changes):
    value = {'kind': 'remote_plan', 'version': 1, 'data_type': 'image_text', 'layout': 'manifest',
             'records': [{'messages': [{'role': 'user', 'content': '<image>Color?'}, {'role': 'assistant', 'content': 'red'}],
                          'images': ['images/a.png']}],
             'files': [{'path': 'images/a.png', 'size': len(png('red'))}], **changes}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(value))
    return path


def test_manifest_layout_downloads_media_from_source_and_validates(tmp_path, storage, monkeypatch):
    files, opened = storage
    files['datasets/receipts/images/a.png'] = png('red')
    monkeypatch.setattr(v2_runtime, '_download', lambda url, target: target.write_bytes(plan(tmp_path).read_bytes()))
    train = tmp_path / 'train'
    train.mkdir()
    data_dir, report = v2_runtime.prepare_dataset(['ft-file://' + 'a' * 64], 'image_text', train, source())
    assert (data_dir / 'images/a.png').read_bytes() == png('red')
    assert report['samples'] == 1 and report['media'] == 1 and report['source'] == 'sftp'
    assert json.loads((data_dir / 'dataset_info.json').read_text())['bisheng_train']['columns']['images'] == 'images'
    assert opened[0]['password'] == SECRET


def test_zip_layout_and_changed_source_file(tmp_path, storage, monkeypatch):
    files, _ = storage
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        archive.writestr('train.json', json.dumps([{'instruction': 'q', 'output': 'a'}]))
    files['datasets/text.zip'] = buffer.getvalue()
    zip_plan = plan(tmp_path, data_type='text', layout='zip', zip={'path': 'datasets/text.zip', 'size': len(buffer.getvalue())})
    monkeypatch.setattr(v2_runtime, '_download', lambda url, target: target.write_bytes(zip_plan.read_bytes()))
    train = tmp_path / 'train'
    train.mkdir()
    _, report = v2_runtime.prepare_dataset(['x'], 'text', train, source())
    assert report['samples'] == 1
    files['datasets/text.zip'] = b'changed'
    other = tmp_path / 'other'
    other.mkdir()
    with pytest.raises(v2_runtime.V2RuntimeError, match='changed after import'):
        v2_runtime.prepare_dataset(['x'], 'text', other, source())


def test_plan_cannot_escape_the_dataset_directory(tmp_path, storage, monkeypatch):
    escape = plan(tmp_path, files=[{'path': '../outside.png', 'size': 1}])
    monkeypatch.setattr(v2_runtime, '_download', lambda url, target: target.write_bytes(escape.read_bytes()))
    train = tmp_path / 'train'
    train.mkdir()
    with pytest.raises(v2_runtime.V2RuntimeError, match='unsafe'):
        v2_runtime.prepare_dataset(['x'], 'image_text', train, source())
    assert not (tmp_path / 'outside.png').exists()


def test_size_limit_and_cancellation_stop_the_download(tmp_path, storage):
    files, _ = storage
    files['datasets/receipts/images/a.png'] = b'12345'
    item = [{'remote': 'datasets/receipts/images/a.png', 'local': 'a.png', 'size': 5}]
    with pytest.raises(remote_source.RemoteSourceError, match='size limit'):
        remote_source.fetch(source(), item, tmp_path / 'a', byte_limit=4)
    with pytest.raises(remote_source.RemoteSourceError, match='cancelled'):
        remote_source.fetch(source(), item, tmp_path / 'b', byte_limit=10, cancelled=lambda: True)


@pytest.mark.parametrize('address', ['127.0.0.1', '169.254.169.254', '::1'])
def test_loopback_and_metadata_endpoints_are_refused(monkeypatch, address):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, '', (address, 22))])
    with pytest.raises(remote_source.RemoteSourceError, match='not allowed'):
        remote_source._endpoint(source())


def test_connector_errors_never_carry_secrets(monkeypatch):
    monkeypatch.setattr(remote_source, '_endpoint', lambda s: (None, 22, '192.168.1.10'))

    class Broken:
        def __init__(self, *args):
            raise OSError(f'auth failed for {SECRET}')

    monkeypatch.setattr(remote_source, '_Sftp', Broken)
    with pytest.raises(remote_source.RemoteSourceError) as raised:
        with remote_source.open_source(source()):
            pass
    assert SECRET not in str(raised.value)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'ROOT', tmp_path / 'jobs')
    monkeypatch.setattr(service, 'MODELS', tmp_path / 'models')
    monkeypatch.setattr(service, 'active_job', None)
    monkeypatch.setattr(service, 'inference_busy', False)
    monkeypatch.setattr(service, 'run_preflight', lambda *args, **kwargs: {'passed': True, 'checks': []})
    with TestClient(service.app) as client:
        yield client


def job(identifier='remote-1', **params):
    return {'job_id': identifier, 'options': ['train'], 'params': {'contract_version': 2, 'capability_id': 'qwen2.5-vl-3b-instruct-lora-20261102',
            'model_name_or_path': 'Qwen2.5-VL-3B-Instruct', 'data_type': 'image_text', 'finetuning_type': 'lora',
            'dataset': 'ft-file://' + 'b' * 64, 'dataset_source': source(), 'max_steps': 2, **params}}


def test_job_keeps_secrets_in_memory_only(client, monkeypatch):
    checked, received = [], threading.Event()
    seen = {}
    monkeypatch.setattr(remote_source, 'check', lambda src, probe: checked.append((src['password'], probe)) or 10)

    def run(identifier, params, job_dir, models, started, cancelled, src):
        seen.update(src=src, params=params)
        received.set()
        raise ValueError('stop after capturing')

    monkeypatch.setattr(service, 'run_v2_job', run)
    assert client.post('/v2.1/sft/job', json=job()).status_code == 200
    assert received.wait(2)
    assert checked == [(SECRET, 'datasets/receipts/images/a.png')]
    assert seen['src']['password'] == SECRET
    assert 'password' not in seen['params']['dataset_source']
    for _ in range(100):
        if service.active_job is None:
            break
        time.sleep(.01)
    saved = (service.ROOT / 'remote-1' / 'service-state.json').read_text()
    assert SECRET not in saved and '"type": "sftp"' in saved


def test_unreachable_source_is_rejected_before_the_job_exists(client, monkeypatch):
    def refuse(src, probe):
        raise remote_source.RemoteSourceError('cannot connect to the SFTP data source; check address, account and network access')

    monkeypatch.setattr(remote_source, 'check', refuse)
    result = client.post('/v2.1/sft/job', json=job('remote-2'))
    assert result.status_code == 422 and 'cannot connect' in result.text and SECRET not in result.text
    assert not (service.ROOT / 'remote-2').exists()
    assert client.post('/v2.1/sft/source/check', json={'source': source(), 'probe': 'x'}).status_code == 422


def test_malformed_requests_do_not_echo_the_body(client):
    result = client.post('/v2.1/sft/job', json={'job_id': 'x', 'params': 'not-a-dict', 'secret': SECRET})
    assert result.status_code == 422 and SECRET not in result.text


def test_plan_with_too_many_files_is_refused_before_downloading(tmp_path, storage, monkeypatch):
    _, opened = storage
    many = plan(tmp_path, files=[{'path': f'images/{index}.png', 'size': 1} for index in range(3)])
    monkeypatch.setattr(v2_runtime, '_download', lambda url, target: target.write_bytes(many.read_bytes()))
    monkeypatch.setenv('BISHENG_FT_MAX_DATASET_FILES', '2')
    train = tmp_path / 'train'
    train.mkdir()
    with pytest.raises(v2_runtime.V2RuntimeError, match='more files'):
        v2_runtime.prepare_dataset(['x'], 'image_text', train, source())
    assert opened == []


@pytest.mark.parametrize(('rows', 'columns'), [
    ([{'instruction': 'q', 'output': 'a'}], {'prompt': 'instruction', 'response': 'output'}),
    ([{'instruction': 'q', 'output': 'a'}, {'instruction': 'q', 'input': 'x', 'output': 'a', 'system': 's'}],
     {'prompt': 'instruction', 'response': 'output', 'query': 'input', 'system': 'system'}),
])
def test_alpaca_rows_without_optional_fields_name_only_present_columns(tmp_path, rows, columns):
    # LlamaFactory raised KeyError: 'input' for rows that only had instruction and output.
    (tmp_path / 'train.json').write_text(json.dumps(rows))
    v2_runtime._write_dataset_info(tmp_path, 'text')
    assert json.loads((tmp_path / 'dataset_info.json').read_text())['bisheng_train']['columns'] == columns
