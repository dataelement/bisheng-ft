FROM nvcr.io/nvidia/pytorch:22.08-py3

ARG PIP_REPO=https://pypi.tuna.tsinghua.edu.cn/simple
ARG EXTR_PIP_REPO="http://public:26rS9HRxDqaVy5T@110.16.193.170:50083/repository/pypi-hosted/simple --trusted-host 110.16.193.170"
ARG BISHENG_FT_VER=0.0.1

# 安装系统库依赖
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y nasm zlib1g-dev libssl-dev libre2-dev libb64-dev locales libsm6 libxext6 libxrender-dev libgl1 tmux git

# Configure language
RUN locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

# Configure timezone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN mkdir -p /opt/bisheng-ft/
WORKDIR /opt/bisheng-ft

# 安装bisheng-ft依赖
RUN ln -s /usr/local/bin/pip3 /usr/bin/pip3.8
RUN pip install --upgrade pip
COPY ./requirements.txt /opt/bisheng-ft
RUN pip install -r requirements.txt -i $PIP_REPO

# 安装bisheng-ft
RUN pip install bisheng-ft==${BISHENG_FT_VER} \
    --extra-index $EXTR_PIP_REPO \
    -i $PIP_REPO

# 下载预置数据集
RUN mkdir -p /opt/bisheng-ft/sft_datasets
COPY ./data/alpaca_data_en_52k.json /opt/bisheng-ft/sft_datasets
COPY ./data/alpaca_data_zh_51k.json /opt/bisheng-ft/sft_datasets
COPY ./docker/datasets_download.sh /opt/bisheng-ft/sft_datasets

# 拷贝sft-server代码
COPY ./src/sft_server /opt/bisheng-ft/sft_server
COPY ./docker/start-sft-server.sh /opt/bisheng-ft/
RUN mkdir -p /opt/bisheng-ft/sft_log /opt/bisheng-ft/finetune_output

EXPOSE 8000
CMD ["sh start-sft-server.sh"]