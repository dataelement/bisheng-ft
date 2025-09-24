ARG BASE_IMAGE=ascendai/cann:8.1.rc1-910b-ubuntu22.04-py3.11
FROM ${BASE_IMAGE}

# Installation arguments
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG EXTRAS=torch-npu,metrics
ARG HTTP_PROXY=""
ARG PYTORCH_INDEX=https://download.pytorch.org/whl/cpu

# Define environments
ENV MAX_JOBS=16
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV http_proxy="${HTTP_PROXY}"
ENV https_proxy="${HTTP_PROXY}"

# Use Bash instead of default /bin/sh
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y curl

# Set the working directory
WORKDIR /app

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip install --no-cache-dir --upgrade pip packaging wheel setuptools

# Install torch-npu
RUN pip uninstall -y torch torchvision torchaudio && \
    pip install --no-cache-dir "torch-npu==2.5.1" "torchvision==0.20.1" --index-url "${PYTORCH_INDEX}"

# Install the requirements
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application into the image
COPY . /app

# Install LLaMA Factory
RUN pip install --no-cache-dir -e ".[${EXTRAS}]" --no-build-isolation


# Set up volumes
#VOLUME [ "/root/.cache/huggingface", "/root/.cache/modelscope", "/app/data", "/app/output" ]

RUN mkdir -p /opt/bisheng-ft/sft_datasets
# 拷贝sft-server代码
COPY src/sft_server /opt/bisheng-ft/sft_server
RUN mkdir -p /opt/bisheng-ft/sft_log /opt/bisheng-ft/finetune_output


ENV SYSTEM_ENV=NPU

# Expose port 7860 for the LLaMA Board
ENV GRADIO_SERVER_PORT 7860
EXPOSE 7860

# Expose port 8000 for the API service
ENV API_PORT 8000
EXPOSE 8000
CMD ["bash start-sft-server.sh"]