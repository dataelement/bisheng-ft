from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


class DatasetPackageError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(name: str) -> bool:
    member = PurePosixPath(name)
    return bool(name) and '\\' not in name and ':' not in name and not member.is_absolute() and '..' not in member.parts


def _read_records(path: Path) -> list[dict[str, Any]]:
    try:
        content = path.read_text(encoding='utf-8')
        records = json.loads(content) if content.lstrip().startswith('[') else [json.loads(line) for line in content.splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetPackageError('dataset is not valid UTF-8 JSON') from exc
    if not isinstance(records, list) or not records or any(not isinstance(row, dict) for row in records):
        raise DatasetPackageError('dataset must be a non-empty list of objects')
    return records


def _validate_messages(record: dict[str, Any], index: int) -> None:
    messages = record.get('messages')
    if not isinstance(messages, list) or not messages:
        raise DatasetPackageError(f'record {index} requires non-empty messages')
    if any(not isinstance(m, dict) or m.get('role') not in {'system', 'user', 'assistant'} or not isinstance(m.get('content'), str) or not m['content'].strip() for m in messages):
        raise DatasetPackageError(f'record {index} has invalid message')
    dialogue = messages[1:] if messages[0]['role'] == 'system' else messages
    if len(dialogue) < 2 or len(dialogue) % 2 or any(m['role'] != ('user' if i % 2 == 0 else 'assistant') for i, m in enumerate(dialogue)):
        raise DatasetPackageError(f'record {index} must alternate user and assistant and end with an answer')


def validate_text_json(source: Path) -> dict[str, Any]:
    records = _read_records(source)
    formats = set()
    for index, record in enumerate(records):
        if any(record.get(field) for field in ('images', 'videos', 'audios')):
            raise DatasetPackageError('text dataset cannot reference media')
        if 'messages' in record:
            formats.add('messages')
            _validate_messages(record, index)
            content = ''.join(m['content'] for m in record['messages'])
        else:
            formats.add('alpaca')
            if not all(isinstance(record.get(key, ''), str) for key in ('instruction', 'input', 'output')) or not record.get('instruction') or not record.get('output'):
                raise DatasetPackageError(f'record {index} requires instruction and output')
            content = ''.join(record.get(k, '') for k in ('instruction', 'input', 'output'))
        if any(f'<{kind}>' in content for kind in ('image', 'video', 'audio')):
            raise DatasetPackageError('text dataset cannot contain media placeholders')
    if len(formats) != 1:
        raise DatasetPackageError('dataset cannot mix Alpaca and messages formats')
    return {'data_type': 'text', 'samples': len(records), 'media': 0, 'sha256': sha256_file(source)}


def validate_media(path: Path, data_type: str) -> None:
    try:
        if data_type == 'image_text':
            with Image.open(path) as image:
                image.verify()
        elif data_type == 'video_text':
            import av
            with av.open(str(path)) as clip:
                if not clip.streams.video or clip.duration is None or clip.duration / av.time_base > 10:
                    raise ValueError('video must contain frames and have duration <= 10 seconds')
                stream = clip.streams.video[0]
                if stream.width * stream.height > 1920 * 1080 or next(clip.decode(video=0), None) is None:
                    raise ValueError('video must be decodable and <= 1080p')
        else:
            raise ValueError('unsupported media type')
    except Exception as exc:
        raise DatasetPackageError(f'invalid media {path.name}: {exc}') from exc


def validate_multimodal_zip(source: Path, destination: Path, data_type: str, max_uncompressed_bytes: int = 20 * 1024 * 1024) -> dict[str, Any]:
    if data_type not in ('image_text', 'video_text'):
        raise DatasetPackageError('unsupported data_type')
    destination.mkdir(parents=True, exist_ok=True)
    try:
        archive = zipfile.ZipFile(source)
    except (OSError, zipfile.BadZipFile) as exc:
        raise DatasetPackageError('dataset is not a valid ZIP') from exc
    with archive:
        members = archive.infolist()
        if len(members) > 5000 or sum(item.file_size for item in members) > max_uncompressed_bytes:
            raise DatasetPackageError('uncompressed dataset exceeds configured limit')
        names = [item.filename for item in members]
        if len(names) != len(set(names)):
            raise DatasetPackageError('duplicate archive members')
        for item in members:
            if not _safe_member(item.filename) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise DatasetPackageError(f'unsafe archive member: {item.filename}')
        archive.extractall(destination)
    manifests = [destination / name for name in ('train.json', 'train.jsonl', 'data.json', 'data.jsonl') if (destination / name).is_file()]
    if len(manifests) != 1:
        raise DatasetPackageError('package must have exactly one root train.json/train.jsonl/data.json/data.jsonl')
    records = _read_records(manifests[0])
    manifest = destination / 'train.json'
    manifest.write_text(json.dumps(records, ensure_ascii=False))
    expected = 'images' if data_type == 'image_text' else 'videos'
    referenced: set[str] = set()
    for index, record in enumerate(records):
        refs = record.get(expected)
        if not isinstance(refs, list) or not refs:
            raise DatasetPackageError(f'record {index} requires {expected}')
        for reference in refs:
            if not isinstance(reference, str) or not _safe_member(reference):
                raise DatasetPackageError(f'record {index} has unsafe media reference')
            path = destination / reference
            if not path.is_file():
                raise DatasetPackageError(f'record {index} references missing media: {reference}')
            if reference not in referenced:
                validate_media(path, data_type)
            referenced.add(reference)
        _validate_messages(record, index)
        for field, kind in [('images', 'image'), ('videos', 'video'), ('audios', 'audio')]:
            media = record.get(field, [])
            count = sum(m['content'].count(f'<{kind}>') for m in record['messages'])
            if not isinstance(media, list) or len(media) != count or (media and field != expected):
                raise DatasetPackageError(f'record {index} media/placeholders mismatch; audio is excluded')
    return {'data_type': data_type, 'samples': len(records), 'media': len(referenced), 'sha256': sha256_file(source), 'manifest': str(manifest)}


def validate_image_text_zip(source: Path, destination: Path, max_uncompressed_bytes: int = 20 * 1024 * 1024) -> dict[str, Any]:
    return validate_multimodal_zip(source, destination, 'image_text', max_uncompressed_bytes)
