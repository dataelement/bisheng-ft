import hashlib

import pytest
import requests
from fastapi.testclient import TestClient

import multimodal_service as service
import uploads
import v2_runtime


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(service, 'ROOT', tmp_path / 'jobs')
    monkeypatch.setattr(service, 'MODELS', tmp_path / 'models')
    monkeypatch.setattr(service, 'active_job', None)
    monkeypatch.setattr(service, 'inference_busy', False)
    with TestClient(service.app) as client:
        yield client


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_push_stores_by_digest_and_reuses_existing_file(client):
    content = b'{"instruction": "a", "output": "b"}'
    first = client.put(f'/v2.1/sft/files/{digest(content)}', content=content)
    assert first.status_code == 200 and first.json()['data'] == {'sha256': digest(content), 'size': len(content), 'stored': True}
    assert (service.ROOT / '_uploads' / digest(content)).read_bytes() == content
    again = client.put(f'/v2.1/sft/files/{digest(content)}', content=b'ignored')
    assert again.json()['data']['stored'] is False


def test_push_rejects_mismatch_oversize_and_bad_reference(client, monkeypatch):
    assert client.put(f'/v2.1/sft/files/{digest(b"other")}', content=b'content').status_code == 422
    monkeypatch.setenv('BISHENG_FT_MAX_PACKAGE_BYTES', '4')
    assert client.put(f'/v2.1/sft/files/{digest(b"12345")}', content=b'12345').status_code == 422
    assert client.put('/v2.1/sft/files/NOT-A-DIGEST', content=b'x').status_code == 422
    assert not [p for p in (service.ROOT / '_uploads').iterdir() if p.name.endswith('.part')]


def test_download_copies_pushed_file_and_reports_missing_one(client, tmp_path):
    content = b'video-bytes'
    client.put(f'/v2.1/sft/files/{digest(content)}', content=content)
    target = tmp_path / 'sample.mp4'
    v2_runtime._download(uploads.SCHEME + digest(content), target)
    assert target.read_bytes() == content
    with pytest.raises(v2_runtime.V2RuntimeError, match='missing'):
        v2_runtime._download(uploads.SCHEME + digest(b'absent'), tmp_path / 'other')


def test_probe_reports_unreachable_storage_without_failing(client, monkeypatch):
    def refused(*args, **kwargs):
        raise requests.ConnectionError('refused')
    monkeypatch.setattr(service.requests, 'get', refused)
    result = client.post('/v2.1/sft/probe', json={'url': 'http://10.0.0.9:9000/bisheng/x?sig=1'})
    assert result.status_code == 200
    assert result.json()['data'] == {'reachable': False, 'status': None, 'reason': 'ConnectionError'}
    assert client.post('/v2.1/sft/probe', json={'url': 'file:///etc/passwd'}).status_code == 422
    assert 'push' in client.get('/health').json()['data']['transfer_modes']
