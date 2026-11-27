"""Exercise real validation, exclusion and cancellation on the FT contract."""
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
    parser=argparse.ArgumentParser()
    parser.add_argument('--service',default='http://127.0.0.1:8000')
    parser.add_argument('--samples',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--finished-job',required=True)
    args=parser.parse_args()
    server=http.server.ThreadingHTTPServer(('127.0.0.1',18098),functools.partial(http.server.SimpleHTTPRequestHandler,directory=args.samples))
    threading.Thread(target=server.serve_forever,daemon=True).start()
    checks=[]
    def payload(**changes):
        params={'contract_version':2,'capability_id':'qwen3-0.6b-lora-20261102','model_name_or_path':'Qwen3-0.6B',
                'data_type':'text','finetuning_type':'lora','dataset':'http://127.0.0.1:18098/text/train/data.json','max_steps':1000}
        params.update(changes)
        return {'job_id':'negative-'+uuid.uuid4().hex[:8],'params':params,'options':['train']}
    for name,changes in [('audio_rejected',{'data_type':'audio'}),('wrong_modality_rejected',{'data_type':'video_text'}),('unknown_parameter_rejected',{'shell':'bad'}),('bad_parameter_type_rejected',{'max_steps':'two'})]:
        result=requests.post(args.service+'/v2.1/sft/job',json=payload(**changes),timeout=30)
        checks.append({'name':name,'http_status':result.status_code,'passed':result.status_code==422})
    active=payload()
    result=requests.post(args.service+'/v2.1/sft/job',json=active,timeout=30)
    result.raise_for_status()
    result=requests.post(args.service+'/v2.1/sft/job',json=payload(),timeout=30)
    checks.append({'name':'concurrent_training_blocked','http_status':result.status_code,'passed':result.status_code==409})
    result=requests.post(args.service+'/v2.1/sft/job/'+args.finished_job+'/infer',json={'prompt':'test'},timeout=30)
    checks.append({'name':'training_inference_exclusion','http_status':result.status_code,'passed':result.status_code==409})
    identifier=active['job_id']
    for _ in range(100):
        status=requests.post(args.service+'/v2.1/sft/job/status',json={'job_id':identifier},timeout=30).json()['data']
        if status.get('pid'):
            break
        time.sleep(.1)
    assert status.get('pid'),status
    requests.post(args.service+'/v2.1/sft/job/cancel',json={'job_id':identifier},timeout=30).raise_for_status()
    for _ in range(100):
        status=requests.post(args.service+'/v2.1/sft/job/status',json={'job_id':identifier},timeout=30).json()['data']
        if status['status']=='CANCELLED':
            break
        time.sleep(.1)
    health=requests.get(args.service+'/health',timeout=30).json()['data']
    checks.append({'name':'real_process_cancelled_and_service_unlocked','job_id':identifier,'status':status['status'],
                   'exit_code':status.get('exit_code'),'passed':status['status']=='CANCELLED' and health['active_job_id'] is None})
    Path(args.output).write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2),flush=True)
    assert all(c['passed'] for c in checks)
    server.shutdown()


if __name__=='__main__':
    main()
