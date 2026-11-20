"""Run real FT contract checks against a deployed service; writes real outputs."""
import argparse
import functools
import http.server
import json
import threading
import time
import uuid
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--service', default='http://127.0.0.1:8000')
    parser.add_argument('--samples', required=True)
    parser.add_argument('--kind', action='append', choices=['text','image','video'])
    parser.add_argument('--output', required=True)
    parser.add_argument('--data-base-url', default='http://127.0.0.1:18099')
    args = parser.parse_args()
    server = http.server.ThreadingHTTPServer(('0.0.0.0', 18099), functools.partial(http.server.SimpleHTTPRequestHandler, directory=args.samples))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    for attempt in range(30):
        try:
            requests.get(args.service+'/health', timeout=3).raise_for_status()
            break
        except requests.RequestException:
            if attempt == 29:
                raise
            time.sleep(1)
    results = []
    for kind, datatype, model, capability in [('text','text','Qwen3-0.6B','qwen3-0.6b-lora-20261102'),
            ('image','image_text','Qwen2.5-VL-3B-Instruct','qwen2.5-vl-3b-instruct-lora-20261102'),
            ('video','video_text','Qwen2.5-VL-3B-Instruct','qwen2.5-vl-3b-instruct-lora-20261102')]:
        if args.kind and kind not in args.kind:
            continue
        identifier = 'formal-' + kind + '-' + uuid.uuid4().hex[:8]
        dataset = f'bisheng-{kind}-smoke.zip'
        params = {'contract_version':2,'capability_id':capability,'data_type':datatype,'finetuning_type':'lora',
                  'model_name_or_path':model,'dataset':args.data_base_url+'/'+dataset,'max_steps':2,
                  'num_train_epochs':1,'cutoff_len':1024,'per_device_train_batch_size':1,'gradient_accumulation_steps':1,'lora_rank':8,'learning_rate':.0001}
        response = requests.post(args.service+'/v2.1/models/sft_elem/infer', json={'uri':'/v2.1/sft/job','job_id':identifier,'options':['train'],'params':params}, timeout=60)
        response.raise_for_status()
        print(f'{identifier} submitted', flush=True)
        for _ in range(180):
            status = requests.post(args.service+'/v2.1/sft/job/status', json={'job_id':identifier}, timeout=30).json()['data']
            if status['status'] != 'RUNNING':
                break
            time.sleep(2)
        result = {'kind':kind,'job_id':identifier,'status':status,'inference':[]}
        results.append(result)
        Path(args.output).write_text(json.dumps(results,ensure_ascii=False,indent=2))
        if status['status'] != 'FINISHED':
            raise RuntimeError(json.dumps(status))
        print(f'{identifier} FINISHED artifacts={status.get("artifacts",{}).get("complete")}',flush=True)
        holdout = json.loads((Path(args.samples)/kind/'holdout'/'data.json').read_text())[0]
        prompt = holdout['messages'][0]['content'].replace('<image>','').replace('<video>','')
        media = (holdout.get('images') or holdout.get('videos') or [None])[0]
        for baseline in (True,False):
            request = {'prompt':prompt,'baseline':baseline,'generation':{'max_new_tokens':160}}
            if media:
                request['media_url'] = args.data_base_url+'/'+kind+'/holdout/'+media
            response = requests.post(args.service+f'/v2.1/sft/job/{identifier}/infer',json=request,timeout=300)
            response.raise_for_status()
            output = response.json()['data']
            result['inference'].append(output)
            Path(args.output).write_text(json.dumps(results,ensure_ascii=False,indent=2))
            print(f'{identifier} baseline={baseline} result={output}',flush=True)
    server.shutdown()


if __name__ == '__main__':
    main()
