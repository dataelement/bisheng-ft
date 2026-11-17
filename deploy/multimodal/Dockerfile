FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
ARG LLAMAFACTORY_REV=c683e59edd8f893abc5e1403184a1d0644817d2a
ENV PYTHONUNBUFFERED=1 PIP_ROOT_USER_ACTION=ignore
RUN apt-get update && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/llamafactory
RUN git init && git remote add origin https://github.com/hiyouga/LlamaFactory.git && git fetch --depth 1 origin ${LLAMAFACTORY_REV} && git checkout --detach FETCH_HEAD
COPY deploy/multimodal/constraints.txt /tmp/constraints.txt
RUN python -m pip install --no-cache-dir --upgrade pip packaging wheel setuptools hatchling editables && python -m pip install --no-cache-dir --no-build-isolation -c /tmp/constraints.txt -e . qwen-vl-utils requests
# Copy only service code: the legacy src/llamafactory must never shadow the pinned trainer.
COPY src/sft_server/ /opt/bisheng-ft/service/
COPY scripts/download_models.py /opt/bisheng-ft/download_models.py
WORKDIR /opt/bisheng-ft/service
ENV BISHENG_FT_MODEL_ROOT=/models BISHENG_FT_OUTPUT_ROOT=/data/jobs
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "multimodal_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
