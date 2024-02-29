# !/bin/bash
pushd /opt/bisheng-rt/sft_server
nohup uvicorn --host 0.0.0.0 --port 8000  --workers 4 main:app > "/opt/bisheng-rt/sft_log/server.log" 2>&1 &
celery -A celery_tasks:app worker -c 4 -l INFO > "/opt/bisheng-rt/sft_log/celery-worker.log" 2>&1
