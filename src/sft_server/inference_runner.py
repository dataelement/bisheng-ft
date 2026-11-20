"""Load the pinned trainer's processor, template and adapter (or freeze/full model) in a fresh process."""
import argparse
import json
import os
from pathlib import Path


def build_prompt(prompt, kind):
    """Add the media placeholder once; prompts copied from training data often carry it already."""
    placeholder = f'<{kind}>' if kind else ''
    return prompt if not placeholder or placeholder in prompt else placeholder + prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--response', required=True)
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text())
    from llamafactory.chat import ChatModel
    import torch
    options = {'model_name_or_path': request['model_path'], 'template': request['template'],
               'finetuning_type': 'lora', 'infer_dtype': 'bfloat16', 'flash_attn': 'sdpa',
               'trust_remote_code': False, 'image_max_pixels': 131072, 'video_max_pixels': 65536,
               'video_fps': 2.0, 'video_maxlen': 8}
    if request.get('adapter'):
        options['adapter_name_or_path'] = request['adapter']
    # A freeze/full model loads from the job directory, so the base model name comes with the request.
    if (request.get('base_model') or Path(request['model_path']).name).startswith('Qwen3-'):
        options['enable_thinking'] = False
    model = ChatModel(options)
    kind = {'image_text': 'image', 'video_text': 'video'}.get(request['data_type'])
    kwargs = {kind + 's': [request['media_path']]} if kind else {}
    prompt = build_prompt(request['prompt'], kind)
    response = model.chat([{'role': 'user', 'content': prompt}], do_sample=False,
                          max_new_tokens=request.get('generation', {}).get('max_new_tokens', 160), **kwargs)[0]
    result = {'completed': True, 'answer': response.response_text, 'finish_reason': response.finish_reason,
              'baseline': request.get('baseline', False), 'pid': os.getpid(),
              'peak_memory_mib': torch.cuda.max_memory_allocated() / 1024**2}
    Path(args.response).write_text(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
