import os
import glob
import shutil
import random
dir_path = os.path.dirname(os.path.realpath(__file__))

model_config = {
    'baichuan2': {'default_module': 'W_pack', 'template': 'baichuan2'},
    'chatglm2': {'default_module': 'query_key_value', 'template': 'chatglm2'},
    'chatglm3': {'default_module': 'query_key_value', 'template': 'chatglm3'},
    'internlm': {'default_module': 'q_proj,v_proj', 'template': 'intern'},
    'llama2': {'default_module': 'q_proj,v_proj', 'template': 'llama2'},
    'qwen': {'default_module': 'c_attn', 'template': 'qwen'},
}

model_name_mapping = {
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
    max_samples = args.max_samples 
    finetuning_type = args.finetuning_type
    per_device_train_batch_size = args.per_device_train_batch_size
    learning_rate = args.learning_rate
    num_train_epochs = args.num_train_epochs
    max_seq_len = args.max_seq_len
    gpus = args.gpus
    cpu_load = args.cpu_load

    if not os.path.exists(model_name_or_path):
        raise ValueError(f'base model path {model_name_or_path} not exists')
    base_model_name = os.path.basename(model_name_or_path)
    if base_model_name not in model_name_mapping.keys():
        raise ValueError(f'base model name {base_model_name} not supported')
    
    base_config = model_config[model_name_mapping[base_model_name]]

    # remain params: lora_target, quantization_bit, max_samples
    train_params_cmd = f'''
--stage sft \
--do_train True \
--finetuning_type {finetuning_type} \
--model_name_or_path {model_name_or_path} \
--template {base_config['template']} \
--dataset {dataset} \
--val_size {val_ratio} \
--output_dir {output_dir} \
--overwrite_output_dir \
--cutoff_len {max_seq_len} \
--learning_rate {learning_rate} \
--per_device_train_batch_size {per_device_train_batch_size} \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 4 \
--lr_scheduler_type cosine \
--num_train_epochs {num_train_epochs} \
--logging_steps 10 \
--save_strategy epoch \
--evaluation_strategy epoch \
--metric_for_best_model eval_loss \
--load_best_model_at_end True \
--save_total_limit 1 \
--fp16 True \
--plot_loss True \
'''
    if max_samples is not None:
        train_params_cmd += f'''--max_samples {max_samples} \\'''
   
    export_params_cmd = f'''
--model_name_or_path {model_name_or_path} \
--template {base_config['template']} \
--finetuning_type lora \
--checkpoint_dir {output_dir} \
--export_dir {output_dir} \
'''
    
    predict_params_cmd = f'''
--stage sft \
--do_predict True \
--finetuning_type full \
--model_name_or_path {model_name_or_path} \
--template {base_config['template']} \
--dataset {dataset} \
--max_samples 100 \
--checkpoint_dir {output_dir} \
--output_dir {output_dir} \
--cutoff_len {max_seq_len} \
--per_device_eval_batch_size 1 \
--predict_with_generate \
--max_new_tokens 512 \
--top_p 0.7 \
--temperature 0.95 \
--fp16 True \
'''

    finetune_file = os.path.join(dir_path, '..', 'tuner', 'tune.py')
    if (len(gpus.split(',')) == 1) and (not cpu_load):
        # Train on a single GPU 
        train_cmd = f'''CUDA_VISIBLE_DEVICES={gpus} python {finetune_file} \\''' + train_params_cmd
        if finetuning_type == 'lora':
            train_cmd += f'''--lora_target {base_config['default_module']}'''
    else:
        if cpu_load:
            deepspeed_file = 'ds_config_zero3_cpu.json'
        else:
            deepspeed_file = 'ds_config_zero2.json'
        master_port_id = random.randint(1000, 9999)
        train_cmd = f'''deepspeed -i localhost:{gpus} --master_port={master_port_id} {finetune_file} --deepspeed {os.path.join(dir_path, deepspeed_file)} \\''' + train_params_cmd
        if finetuning_type == 'lora':
            train_cmd += f'''--lora_target {base_config['default_module']}'''

    # phase1: train, print train loss and eval loss, train log saved in trainer_log.jsonl    
    print('train_cmd:', train_cmd)
    os.system(train_cmd)
    checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
    for checkpoint in checkpoints:
        shutil.rmtree(checkpoint)
    
    # phase2: merge LoRA weights and export model, generate pytorch_model-0000*.bin
    if finetuning_type == 'lora':
        export_file = os.path.join(dir_path, 'export_model.py')
        export_cmd = f'''python {export_file} \\''' + export_params_cmd 
        os.system(export_cmd)

    # phase3: predict 100 example, compute metrics (ROUGE, BLEU), metrics saved in predict_results.json, predictions saved in generated_predictions.jsonl
    # todo: only support single gpu predict(multi gpu deepspeed infer is so slow)
    predict_cmd = f'''CUDA_VISIBLE_DEVICES={gpus.split(',')[0]} python {finetune_file} \\''' + predict_params_cmd
    print('predict_cmd:', predict_cmd)
    os.system(predict_cmd)
    
