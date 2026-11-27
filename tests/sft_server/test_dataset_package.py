import io
import json
import zipfile

import pytest
from dataset_package import DatasetPackageError, validate_image_text_zip, validate_text_json
from PIL import Image


def _png_bytes():
    stream = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(stream, format="PNG")
    return stream.getvalue()


def _zip(path, records, extra=None):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("train.json", json.dumps(records))
        archive.writestr("media/sample.png", _png_bytes())
        for name, value in extra or []:
            archive.writestr(name, value)


def test_text_alpaca_dataset(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps([{"instruction": "a", "input": "", "output": "b"}]), encoding="utf-8")
    assert validate_text_json(path)["samples"] == 1


def test_image_text_package(tmp_path):
    package = tmp_path / "data.zip"
    records = [
        {
            "messages": [{"role": "user", "content": "<image>what"}, {"role": "assistant", "content": "ok"}],
            "images": ["media/sample.png"],
        }
    ]
    _zip(package, records)
    report = validate_image_text_zip(package, tmp_path / "out")
    assert report["samples"] == 1
    assert report["media"] == 1


def test_zip_path_traversal_is_rejected(tmp_path):
    package = tmp_path / "bad.zip"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("../escape", "bad")
    with pytest.raises(DatasetPackageError, match="unsafe archive"):
        validate_image_text_zip(package, tmp_path / "out")


def test_missing_media_is_rejected(tmp_path):
    package = tmp_path / "bad.zip"
    records = [{"messages": [{"role": "user", "content": "x"}], "images": ["media/missing.png"]}]
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("train.json", json.dumps(records))
    with pytest.raises(DatasetPackageError, match="missing media"):
        validate_image_text_zip(package, tmp_path / "out")


def test_duplicate_archive_and_placeholder_mismatch_are_rejected(tmp_path):
    package = tmp_path / 'data.zip'
    records = [{'messages':[{'role':'user','content':'no placeholder'}, {'role':'assistant','content':'answer'}], 'images':['media/sample.png']}]
    _zip(package, records)
    with pytest.raises(DatasetPackageError, match='placeholders mismatch'):
        validate_image_text_zip(package, tmp_path/'out')


def test_text_audio_is_rejected(tmp_path):
    path = tmp_path / 'train.json'
    path.write_text(json.dumps([{'messages':[{'role':'user','content':'<audio>transcribe'}, {'role':'assistant','content':'hello'}], 'audios':['clip.wav']}]))
    with pytest.raises(DatasetPackageError, match='cannot reference media'):
        validate_text_json(path)
