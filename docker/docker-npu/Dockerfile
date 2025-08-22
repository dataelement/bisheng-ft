# Use the Ubuntu 22.04 image with CANN 8.0.rc1
# More versions can be found at https://hub.docker.com/r/ascendai/cann/tags
# FROM ascendai/cann:8.0.rc1-910-ubuntu22.04-py3.8
FROM ascendai/cann:8.1.rc1-910b-ubuntu22.04-py3.11
# FROM ascendai/cann:8.0.rc1-910-openeuler22.03-py3.8
# FROM ascendai/cann:8.0.rc1-910b-openeuler22.03-py3.8

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

# Define installation arguments
ARG INSTALL_DEEPSPEED=false
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# Use Bash instead of default /bin/sh
SHELL ["/bin/bash", "-c"]

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

# Expose port 7860 for the LLaMA Board
ENV GRADIO_SERVER_PORT 7860
EXPOSE 7860

# Expose port 8000 for the API service
ENV API_PORT 8000
EXPOSE 8000
CMD ["sh start-sft-server.sh"]