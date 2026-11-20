# 毕昇 FT：文本、图文与视频

此部署运行正式 `/v2.1/sft` 契约，使用固定 LlamaFactory `c683e59edd8f893abc5e1403184a1d0644817d2a`。单服务管理一张分配的 NVIDIA GPU，同一时刻执行一项训练或推理；无需 Redis。保留的旧 `main.py` / Celery 部署是另一入口，勿与此服务混用。

## 安装

前提：Linux 服务器、可用 NVIDIA 驱动、Docker Engine、Docker Compose 插件以及 NVIDIA Container Toolkit；`nvidia-smi` 正常，Docker 可访问管理员分配的 GPU。开始小模型 LoRA 建议 24 GB 显存和 40 GB 可用磁盘，实际需求取决于模型、长度及媒体分辨率。

在此目录执行：

```bash
cp .env.example .env
# 编辑 .env：GPU_ID 选已分配的卡；FT_BIND_IP 填服务器内网 IP；确认存储目录。
mkdir -p data/models data/jobs
docker compose build ft
docker compose --profile tools run --rm download
docker compose up -d ft
docker compose ps
```

首次下载需要访问所配置的 Hugging Face 端点。断开后重新运行相同命令可续传，脚本固定权重修订并验证 SHA-256。仅安装文本模型可执行 `docker compose --profile tools run --rm download --model Qwen3-0.6B`。命令行不应包含账号密码；私有仓库访问凭证由企业管理员单独配置。

使用服务器实际内网地址检查 `/health`、`/v2.1/sft/model`、`/v2.1/sft/capabilities`；状态码 200 且模型 `ready=true` 后，在毕昇「模型微调」登记该 HTTP 地址。Bisheng 后端必须能访问 FT。服务只应开放给可信内网的平台后端；本服务不提供公网登录入口，接口本身不做身份校验。

训练数据和验证图片 / 视频有两种送达方式，由平台每次提交时自动选择：FT 能下载平台存储链接时由 FT 下载（拉取）；下载不到时由平台推送给 FT（直传，`PUT /v2.1/sft/files/{sha256}`）。所以 FT 访问不到毕昇的对象存储也能用，不必为此单独开放 GPU 服务器到毕昇的访问。直传的文件按内容哈希保存在 `OUTPUT_DIR/_uploads/`，删除任务不会清理，磁盘紧张时在没有任务运行时手动清理。

训练方式：文本模型支持 LoRA、Freeze（只训练最后几层，`freeze_trainable_layers`）和全量；图文、视频模型支持 LoRA 和 Freeze，全量训练单张 24 GB 显卡放不下，能力目录不提供。Freeze 和全量训练产出的是完整模型（Qwen3-0.6B 全量约 2.3 GB、Freeze 约 1.2 GB），新样本推理直接加载任务目录里的模型。

从 FTP / FTPS / SFTP / S3 导入的数据集不经过毕昇：平台只登记清单和每个文件的大小，开始训练时 FT 用本次提交带来的账号密码直接从数据源下载，并核对大小与导入时一致。所以 **GPU 服务器必须能访问该数据源**。接任务前 FT 会先登录数据源并读取一个文件（`POST /v2.1/sft/source/check`），失败时任务不会创建。密码和密钥只在内存里用于这一次下载，不写入任务目录、状态文件或日志。单个数据集默认最多 200 GiB、200,000 个文件，可用 `BISHENG_FT_MAX_DATASET_BYTES`、`BISHENG_FT_MAX_DATASET_FILES` 调整；下载前会检查磁盘至少留出数据集大小加 1 GiB 的空间。平台直传或拉取的单个数据包上限为 2 GiB（`BISHENG_FT_MAX_PACKAGE_BYTES`）。

## 登记和分配模型

`.env` 的 `MODEL_DIR` 挂载基座权重，模型目录名必须与能力目录 `model_name` 相同。`CAPABILITIES_FILE` 指向 JSON 能力目录，包含模型名、模板、输入类型、方法、参数范围和验证状态；默认读取随代码提供的目录。`FT_MODELS` 仅筛选本服务开放的登记模型，不能让未知模型自动获得训练能力。

新模型由管理员下载完整权重，登记模型架构对应的训练模板，再验证真实训练与重载；不得只修改 `validation_status` 来跳过验证。向客户开放前应固定模型修订及运行环境。多个服务使用独立输出目录和端口，可共享只读权重，每个服务设置自己的 `FT_MODELS`。更改 `.env` 后执行 `docker compose up -d ft`，再在平台刷新服务模型。

## 数据与训练

文本支持 UTF-8 JSON 数组或 JSONL（Alpaca 或 messages，单文件内格式一致），也支持含单个清单的 ZIP。图文/视频使用 ZIP：根目录一个 `train.json`、`train.jsonl`、`data.json` 或 `data.jsonl`，媒体以相对路径引用；图文用 `images` 和 `<image>`，视频用 `videos` 和 `<video>`。每条对话以用户提问和助手标准答案成对出现；音频不支持。ZIP 解压上限 20 MiB，视频不超过 10 秒和 1080p，处理最多 8 帧。

训练参数来自平台任务，并保存为实际 `train.yaml`。`max_steps` 为正数时优先于轮数；填 `-1` 时按 `num_train_epochs` 训练。当前只开放 LoRA；图文与视频冻结视觉编码器及投影层，在语言部分训练适配器。任务状态、数据哈希、训练日志、产物哈希和新进程推理输出保存在 `OUTPUT_DIR` 的任务子目录。基座权重为只读，adapter 需与对应基座一起重载。

## 运维边界

关闭或重启前确认没有训练或推理运行。异常重启时未完成任务标记为 interrupted，不自动重复提交；先核查日志和模型产物，再从平台重建任务。重复任务 ID 返回 409，避免重复训练。取消会终止该任务的进程组并等待真实退出；不会终止别的服务。

产物验证页可测试微调结果并与基础模型比较，推理完成不代表答案正确。模型上线推理服务属于独立部署流程，本服务不会把 adapter 移动到基座模型目录或宣称自动发布完成。

## 开发验证

```bash
python -m pytest tests/sft_server -q
```

`scripts/verify_three_modal.py` 使用真实 FT 端点执行文本/图文/视频 LoRA 与新样本重载。需要准备兼容样本目录和可从 FT 访问的数据 HTTP 地址；此脚本会占用 GPU。
