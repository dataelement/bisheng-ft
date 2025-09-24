
FROM hiyouga/pytorch:th2.6.0-cu124-flashattn2.7.4-cxx11abi0-devel

# Installation arguments
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG EXTRAS=metrics
ARG INSTALL_FLASHATTN=false

# Define environments
ENV MAX_JOBS=16
ENV FLASH_ATTENTION_FORCE_BUILD=TRUE
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV DEBIAN_FRONTEND=noninteractive
ENV NODE_OPTIONS=""
ENV PIP_ROOT_USER_ACTION=ignore
ENV SYSTEM_ENV=CUDA

# Use Bash instead of default /bin/sh
SHELL ["/bin/bash", "-c"]

# install curl
RUN apt-get update && apt-get install -y curl

# Set the working directory
WORKDIR /app

# Change pip source
RUN pip config set global.index-url "${PIP_INDEX}" && \
    pip config set global.extra-index-url "${PIP_INDEX}" && \
    pip install --no-cache-dir --upgrade pip packaging wheel setuptools

# Install the requirements
COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application into the image
COPY . /app

# Install LLaMA Factory
RUN pip install --no-cache-dir -e ".[${EXTRAS}]" --no-build-isolation

# Rebuild flash attention
RUN if [ "${INSTALL_FLASHATTN}" == "true" ]; then \
        pip uninstall -y ninja && \
        pip install --no-cache-dir ninja && \
        pip install --no-cache-dir flash-attn --no-build-isolation; \
    fi

# Expose port 8000 for the API service
ENV API_PORT 8000
EXPOSE 8000
RUN mkdir -p /opt/bisheng-ft/sft_datasets
# 拷贝sft-server代码
COPY src/sft_server /opt/bisheng-ft/sft_server
RUN mkdir -p /opt/bisheng-ft/sft_log /opt/bisheng-ft/finetune_output
CMD ["bash start-sft-server.sh"]