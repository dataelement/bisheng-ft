import json
import os
import subprocess
import sys
from logging import getLogger

import uvicorn
from fastapi.responses import PlainTextResponse, ORJSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi import FastAPI, HTTPException, Request, status, Body

from sft_manage import SFTManage
from celery_tasks import create_job_task
import settings

logger = getLogger(__name__)


def handle_http_exception(req: Request, exc: HTTPException) -> ORJSONResponse:
    msg = {'status_code': exc.status_code, 'status_message': exc.detail}
    logger.error(f'{req.method} {req.url} {exc.status_code} {exc.detail}')
    return ORJSONResponse(content=msg)


def handle_request_validation_error(req: Request, exc: RequestValidationError) -> ORJSONResponse:
    msg = {'status_code': status.HTTP_422_UNPROCESSABLE_ENTITY, 'status_message': exc.errors()}
    logger.error(f'{req.method} {req.url} {exc.errors()} {exc.body}')
    return ORJSONResponse(content=msg)


_EXCEPTION_HANDLERS = {HTTPException: handle_http_exception, RequestValidationError: handle_request_validation_error}

app = FastAPI(
    default_response_class=ORJSONResponse,
    exception_handlers=_EXCEPTION_HANDLERS,
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/health")
def health():
    return {"status_code": 200, "status_message": "success"}


@app.post("/v2.1/sft/job")
def create_job(job_id: str = Body(), options: list = Body(), params: dict = Body()):
    logger.info(f'start create job: {job_id}')
    create_job_task.delay(job_id, options, params)
    return {"status_code": 200, "status_message": "success"}


@app.post("/v2.1/sft/job/delete")
@app.delete("/v2.1/sft/job/delete")
def delete_model(job_id: str = Body(), model_name: str = Body()):
    logger.info(f'delete job: {job_id}')
    sft_obj = SFTManage(job_id)
    sft_obj.delete_job(model_name)
    return {"status_code": 200, "status_message": "success"}


@app.post("/v2.1/sft/job/model_name")
def change_model_name(job_id: str = Body(), old_model_name: str = Body(), model_name: str = Body()):
    logger.info(f'change model name job: {job_id}')
    sft_obj = SFTManage(job_id)
    sft_obj.change_model_name(old_model_name, model_name)
    return {"status_code": 200, "status_message": "success", "data": {"model_name": model_name}}


@app.post("/v2.1/sft/job/cancel")
def cancel_job(job_id: str = Body()):
    logger.info(f'cancel job: {job_id}')
    sft_obj = SFTManage(job_id)
    sft_obj.cancel_job()
    return {"status_code": 200, "status_message": "success"}


@app.post("/v2.1/sft/job/publish")
def publish_model(job_id: str = Body(), model_name: str = Body()):
    logger.info(f'publish model job: {job_id}')
    sft_obj = SFTManage(job_id)
    sft_obj.publish_job(model_name)
    return {"status_code": 200, "status_message": "success"}


@app.post('v2.1/sft/job/publish/cancel')
def cancel_publish_model(job_id: str = Body(), model_name: str = Body()):
    logger.info(f'cancel publish model job: {job_id}')
    sft_obj = SFTManage(job_id)
    sft_obj.cancel_publish_job(model_name)
    return {"status_code": 200, "status_message": "success"}


@app.post("/v2.1/sft/job/status")
@app.get("/v2.1/sft/job/status")
def get_job_status(job_id: str = Body()):
    logger.info(f'get job status job: {job_id}')
    sft_obj = SFTManage(job_id)
    status, reason = sft_obj.get_job_status()
    return {"status_code": 200, "status_message": "success", "data": {
        "status": status,
        "reason": reason
    }}


@app.post("/v2.1/sft/job/log")
@app.get("/v2.1/sft/job/log")
def get_job_log(job_id: str = Body()):
    logger.info(f'get job log job: {job_id}')
    sft_obj = SFTManage(job_id)
    log = sft_obj.get_job_log()
    return {"status_code": 200, "status_message": "success", "data": {
        "log_data": log
    }}


@app.post("/v2.1/sft/job/metrics")
@app.get("/v2.1/sft/job/metrics")
def get_job_report(job_id: str = Body()):
    logger.info(f'get job report job: {job_id}')
    sft_obj = SFTManage(job_id)
    metrics = sft_obj.get_job_metrics()
    return {"status_code": 200, "status_message": "success", "data": {
        "report": metrics
    }}


@app.get('/v2.1/sft/model')
def get_all_model():
    model_list = SFTManage.get_all_model()
    return {
        "status_code": 200,
        "status_message": "success",
        "data": model_list
    }


@app.get('/v2.1/sft/gpu')
def get_gpu_info():
    logger.info('get gpu info')
    p = subprocess.run('nvidia-smi -q -x', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return {"status_code": 400, "status_message": f'fail to get gpu info {p.stderr}'}
    return {"status_code": 200, "status_message": "success", "data": p.stdout.decode('utf-8')}


@app.post("/v2.1/models/sft_elem/infer")
def common_sft_req(payload: dict):
    logger.info(f'receive request: {payload}')
    if payload['uri'] == '/v2.1/sft/job':
        return create_job(payload['job_id'], payload['options'], payload['params'])
    elif payload['uri'] == '/v2.1/sft/job/cancel':
        return cancel_job(payload['job_id'])
    elif payload['uri'] == '/v2.1/sft/job/delete':
        return delete_model(payload['job_id'], payload['model_name'])
    elif payload['uri'] == '/v2.1/sft/job/model_name':
        return change_model_name(payload['job_id'], payload['old_model_name'], payload['model_name'])
    elif payload['uri'] == '/v2.1/sft/job/publish':
        return publish_model(payload['job_id'], payload['model_name'])
    elif payload['uri'] == '/v2.1/sft/job/publish/cancel':
        return cancel_publish_model(payload['job_id'], payload['model_name'])
    elif payload['uri'] == '/v2.1/sft/job/status':
        return get_job_status(payload['job_id'])
    elif payload['uri'] == '/v2.1/sft/job/log':
        return get_job_log(payload['job_id'])
    elif payload['uri'] == '/v2.1/sft/job/metrics':
        return get_job_report(payload['job_id'])

    return {"status_code": 404, "status_message": f"not found handler {payload['uri']}"}


if __name__ == '__main__':
    port = 8000
    if sys.argv.__len__() >= 2:
        port = int(sys.argv[1])
    uvicorn.run(app, host="0.0.0.0", port=port)
