"""Download registered model weights at pinned revisions into the customer model volume."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='append', choices=['Qwen3-0.6B', 'Qwen2.5-VL-3B-Instruct'])
    parser.add_argument('--model-root', default='/models')
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    revisions = {'Qwen3-0.6B': 'c1899de289a04d12100db370d81485cdf75e47ca',
                 'Qwen2.5-VL-3B-Instruct': '66285546d2b821cf421d4f5eb2576359d3770cd3'}
    for model in args.model or list(revisions):
        root = Path(args.model_root) / model
        revision = revisions[model]
        snapshot_download('Qwen/' + model, revision=revision, local_dir=str(root),
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model'])
        metadata = HfApi().model_info('Qwen/' + model, revision=revision, files_metadata=True)
        files = []
        for item in metadata.siblings:
            if not item.rfilename.endswith('.safetensors'):
                continue
            path = root / item.rfilename
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            checksum = digest.hexdigest()
            if not item.lfs or checksum != item.lfs.sha256:
                raise RuntimeError(f'weight checksum failed: {item.rfilename}')
            files.append({'name': item.rfilename, 'sha256': checksum, 'size': path.stat().st_size})
        if not files:
            raise RuntimeError('model contains no verified weights')
        (root / 'bisheng-download-manifest.json').write_text(json.dumps({'model': model, 'revision': revision, 'files': files}, indent=2))
        print(f'{model}: downloaded and verified {len(files)} weight files', flush=True)


if __name__ == '__main__':
    main()
