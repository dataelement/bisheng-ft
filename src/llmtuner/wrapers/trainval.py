import os
import glob
import shutil
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))

model_config = {
    'baichuan': {'default_module': 'W_pack', 'template': 'baichuan'},
    'baichuan2': {'default_module': 'W_pack', 'template': 'baichuan2'},
    'chatglm2': {'default_module': 'query_key_value', 'template': 'chatglm2'},
    'chatglm3': {'default_module': 'query_key_value', 'template': 'chatglm3'},
    'internlm': {'default_module': 'q_proj,v_proj', 'template': 'intern'},
    'llama2': {'default_module': 'q_proj,v_proj', 'template': 'llama2'},
    'qwen': {'default_module': 'c_attn', 'template': 'qwen'},
}

model_name_mapping = {
    'Baichuan-7B-Chat': 'baichuan', 
    'Baichuan-13B-Chat': 'baichuan', 
    'Baichuan2-7B-Chat': 'baichuan2', 
    'Baichuan2-13B-Chat': 'baichuan2',
    'chatglm2-6b': 'chatglm2',
    'chatglm3-6b': 'chatglm3',
    'internlm-chat-7b-8k': 'internlm',
    'internlm-20b-chat': 'internlm',
    'Llama-2-7b-chat-hf': 'llama2',
    'Llama-2-13b-chat-hf': 'llama2',
    'Qwen-7B-Chat': 'qwen',
    'Qwen-14B-Chat': 'qwen',
}

def trval_main(args):
    """train and val"""
    model_name_or_path = args.model_name_or_path
    dataset = args.dataset
    output_dir = args.output_dir
    val_ratio = args.val_ratio
    finetuning_type = args.finetuning_type
    per_device_train_batch_size = args.per_device_train_batch_size
    learning_rate = args.learning_rate
    num_train_epochs = args.num_train_epochs
    max_seq_len = args.max_seq_len
    gpus = args.gpus
    cpu_load = args.cpu_load
    finetune_file = os.path.join(dir_path, '..', 'tuner', 'tune.py')

    if not os.path.exists(model_name_or_path):
        raise ValueError(f'base model path {model_name_or_path} not exists')
    base_model_name = os.path.basename(model_name_or_path)
    if base_model_name not in model_name_mapping.keys():
        raise ValueError(f'base model name {base_model_name} not supported')
    
    base_config = model_config[model_name_mapping[base_model_name]]

    # remain params: lora_target, quantization_bit, max_samples
    params_cmd = f'''
--stage sft \
--do_train True \
--do_eval False \
--finetuning_type {finetuning_type} \
--model_name_or_path {model_name_or_path} \
--template {base_config['template']} \
--dataset_dir data \
--dataset {dataset} \
--val_size {val_ratio} \
--output_dir {output_dir} \
--overwrite_output_dir \
--cutoff_len {max_seq_len} \
--learning_rate {learning_rate} \
--per_device_train_batch_size {per_device_train_batch_size} \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--num_train_epochs {num_train_epochs} \
--logging_steps 10 \
--save_strategy no \
--save_steps 500 \
--evaluation_strategy no \
--eval_steps 500 \
--plot_loss True \
--load_best_model_at_end False \
--fp16 True \
'''
    
    if len(gpus.split(',')) == 1 and (not cpu_load):
        # Train on a single GPU 
        train_cmd = f'''CUDA_VISIBLE_DEVICES={gpus} python {finetune_file} \\''' + params_cmd
        if finetuning_type == 'lora':
            train_cmd += f'''--lora_target {base_config['default_module']}'''
    else:
        if cpu_load:
            deepspeed_file = 'ds_config_zero3_cpu.json'
        else:
            deepspeed_file = 'ds_config_zero2.json'
        train_cmd = f'''deepspeed -i localhost:{gpus} --master_port={os.getpid()+1000} {finetune_file} --deepspeed {os.path.join(dir_path, deepspeed_file)} \\''' + params_cmd
        if finetuning_type == 'lora':
            train_cmd += f'''--lora_target {base_config['default_module']}'''

    os.system(train_cmd)