import glob
import json
import os
import shutil
import sys
from typing import Literal, Optional

from loguru import logger
from pydantic import BaseModel, Field

dir_path = os.path.dirname(os.path.realpath(__file__))


class BishengFTrainArgs(BaseModel):
    # 基础训练设置
    stage: Literal["sft", "rm", "ppo", "dpo", "kto", "pt"] = Field(
        default="sft",
        description="训练阶段: sft(有监督微调), rm(奖励模型训练), ppo(强化学习微调), "
                    "dpo(直接偏好优化), kto(知识蒸馏优化), pt(预训练继续训练)"
    )
    do_train: bool = Field(default=True, description="启用训练模式")

    # 模型配置
    model_name_or_path: str = Field(..., description="预训练模型名称或本地路径")
    cache_dir: Optional[str] = Field(
        default=None,
        description="模型缓存目录, 默认为None使用默认缓存路径"
    )

    trust_remote_code: bool = Field(
        default=True,
        description="是否信任模型库中的自定义代码"
    )

    # 数据处理设置
    dataset_dir: str = Field(..., description="数据集存放根目录")
    dataset: str = Field(..., description="要使用的数据集名称")
    preprocessing_num_workers: int = Field(default=16, description="数据预处理的并行进程数")
    cutoff_len: int = Field(default=2048, description="文本序列的最大长度(token数)")
    max_samples: Optional[str] = Field(
        default=None,
        description="每个数据集的最大样本数, 格式为 'num1,num2,...', "
                    "例如 '1000,2000' 表示第一个数据集使用1000个样本，第二个数据集使用2000个样本。"
                    "如果为None则使用全部样本"
    )
    template: Optional[str] = Field(None, description="对话模板名称")
    packing: bool = Field(default=False, description="是否启用样本打包功能")
    enable_thinking: bool = Field(default=False, description="是否启用思考链格式")

    # 训练超参数
    learning_rate: float = Field(default=5e-5, description="学习率")
    num_train_epochs: float = Field(default=3.0, description="训练轮数")
    per_device_train_batch_size: int = Field(default=2, description="每个设备的训练批次大小")
    gradient_accumulation_steps: int = Field(default=8, description="梯度累积步数")
    lr_scheduler_type: Literal["cosine", "linear", "polynomial", "constant"] = Field(default="cosine",
                                                                                     description="学习率调度策略")
    warmup_steps: int = Field(default=0, description="学习率热身步数")
    max_grad_norm: float = Field(default=1.0, description="梯度裁剪的最大范数")
    optim: Literal["adamw_torch", "adamw_hf", "sgd"] = Field(
        default="adamw_torch",
        description="优化器类型"
    )

    # LoRA参数高效微调设置
    finetuning_type: Literal["lora", "full", "freeze"] = Field(
        default="lora",
        description="微调方式: lora(低秩适应), full(全参数微调), freeze(冻结部分参数)"
    )
    lora_rank: int = Field(default=8, description="LoRA低秩矩阵的维度")
    lora_alpha: int = Field(default=16, description="LoRA的缩放因子")
    lora_dropout: float = Field(default=0.0, description="LoRA层的dropout率")
    lora_target: str = Field(default="all", description="LoRA作用的目标层")

    # 日志与保存设置
    logging_steps: int = Field(default=5, description="日志记录间隔步数")
    save_steps: int = Field(default=100, description="模型保存间隔步数")
    output_dir: str = Field(..., description="训练结果输出目录")
    report_to: Literal["none", "tensorboard", "wandb"] = Field(
        default="none",
        description="训练指标报告工具"
    )
    plot_loss: bool = Field(default=True, description="是否生成损失曲线")

    # 硬件与分布式设置
    fp16: bool = Field(default=True, description="是否启用fp16")
    flash_attn: Literal["auto", "disabled", "sdpa", "fa2"] = Field(
        default="auto",
        description="是否启用Flash Attention加速"
    )
    ddp_timeout: int = Field(default=180000000, description="分布式训练超时时间(毫秒)")
    include_num_input_tokens_seen: bool = Field(
        default=True,
        description="是否记录已处理的token总数"
    )

    # 验证设置
    val_size: float = Field(default=0.0, description="验证集占训练数据的比例")
    eval_strategy: Literal["no", "steps", "epochs"] = Field(
        default="no",
        description="验证策略: steps(按步数), epochs(按轮次)"
    )
    eval_steps: Optional[int] = Field(default=None, description="验证间隔步数")
    per_device_eval_batch_size: int = Field(default=2, description="每个设备的验证批次大小")


def parse_args(args):
    model_name_or_path = args.model_name_or_path
    model_template = args.model_template
    dataset: str = args.dataset
    output_dir = args.output_dir
    val_ratio = args.val_ratio
    each_max_samples = args.each_max_samples
    finetuning_type = args.finetuning_type
    per_device_train_batch_size = args.per_device_train_batch_size
    learning_rate = args.learning_rate
    num_train_epochs = args.num_train_epochs
    max_seq_len = args.max_seq_len
    cpu_load = args.cpu_load

    if not os.path.exists(model_name_or_path):
        raise ValueError(f'base model path {model_name_or_path} not exists')

    from llamafactory.extras.constants import DEFAULT_TEMPLATE
    if model_template not in DEFAULT_TEMPLATE.keys():
        raise ValueError(f'model template {model_template} not supported')

    template = DEFAULT_TEMPLATE[model_template]

    data_dir = dataset.split(',')[0].rsplit("/", 1)[0]
    datasets = dataset.replace(data_dir + "/", "")
    dataset_info = {key: {"file_name": key} for key in datasets.split(',')}
    with open(os.path.join(data_dir, 'dataset_info.json'), "w") as f:
        f.write(json.dumps(dataset_info, indent=4))

    if each_max_samples is not None:
        if len(each_max_samples.split(',')) != len(dataset.split(',')):
            raise ValueError(f'{each_max_samples} and {dataset} should have the same num.')

    bisheng_ft_args = BishengFTrainArgs(
        stage="sft",
        do_train=True,
        model_name_or_path=model_name_or_path,
        dataset_dir=data_dir,
        dataset=datasets,
        template=template,
        cutoff_len=max_seq_len,
        max_samples=each_max_samples,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        finetuning_type=finetuning_type,
        output_dir=output_dir,
        val_size=val_ratio
    )

    return bisheng_ft_args


def trval_main(args):
    bisheng_ft_args = parse_args(args)

    sys.argv = [sys.argv[0], "train"]

    # 将BishengFTrainArgs的字段转换为微调脚本的命令行参数
    for key, value in bisheng_ft_args.model_dump().items():
        if value is not None:
            if isinstance(value, bool):
                if value:
                    sys.argv.append(f'--{key}')
                    sys.argv.append('True')
            else:
                sys.argv.append(f'--{key}')
                sys.argv.append(str(value))

    logger.info('Starting training with args: ' + ' '.join(sys.argv))

    # 第一步：微调训练
    from llamafactory.cli import main as cli_main
    cli_main()

    # 删除中间检查点，节省存储空间
    checkpoints = glob.glob(os.path.join(bisheng_ft_args.output_dir, 'checkpoint-*'))
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)

    # 第二步：如果是LoRA微调，合并Weights并导出模型
    if bisheng_ft_args.finetuning_type == 'lora':
        sys.argv = [sys.argv[0], "export"]
        sys.argv.extend([
            '--model_name_or_path', bisheng_ft_args.model_name_or_path,  # 预训练模型路径
            '--adapter_name_or_path', bisheng_ft_args.output_dir,  # LoRA微调的输出目录
            '--template', bisheng_ft_args.template,
            '--trust_remote_code', str(bisheng_ft_args.trust_remote_code).lower(),
            '--export_dir', bisheng_ft_args.output_dir,  # 导出模型的目录
            '--export_size', '5',  # 导出模型切片大小 单位为GB
            '--export_device', 'cpu'
        ])

        logger.info('Exporting LoRA merged model: ' + ' '.join(sys.argv))
        cli_main()

        # 删除LoRA相关文件，节省存储空间
        os.remove(os.path.join(bisheng_ft_args.output_dir, 'adapter_config.json')) if os.path.exists(
            os.path.join(bisheng_ft_args.output_dir, 'adapter_config.json')) else None
        os.remove(os.path.join(bisheng_ft_args.output_dir, 'adapter_model.safetensors')) if os.path.exists(
            os.path.join(bisheng_ft_args.output_dir, 'adapter_model.safetensors')) else None

    # 第三步：预测和评估
    sys.argv = [sys.argv[0], "train"]
    sys.argv.extend([
        '--stage', 'sft',
        '--do_predict', 'True',
        '--finetuning_type', 'full',
        '--model_name_or_path', bisheng_ft_args.output_dir,  # 使用微调后的模型
        '--template', bisheng_ft_args.template,
        '--eval_dataset', bisheng_ft_args.dataset,
        '--dataset_dir', bisheng_ft_args.dataset_dir,
        '--max_samples', '100',  # 预测100个样本
        '--output_dir', bisheng_ft_args.output_dir,
        '--cutoff_len', str(bisheng_ft_args.cutoff_len),
        '--overwrite_cache', 'True',
        '--per_device_eval_batch_size', '1',
        '--predict_with_generate',
        '--ddp_timeout', str(bisheng_ft_args.ddp_timeout),
        '--max_new_tokens', '512',
        '--top_p', '0.7',
        '--temperature', '0.95',
        '--fp16'
    ])

    logger.info('Starting prediction with args: ' + ' '.join(sys.argv))
    cli_main()
