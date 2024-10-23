run_internlm_20b_qlora4() {
    CUDA_VISIBLE_DEVICES=1 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/public/llm/internlm-20b-chat/ \
    --do_train True \
    --finetuning_type lora \
    --template intern \
    --lora_target q_proj,v_proj \
    --dataset_dir data \
    --dataset alpaca_zh \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --output_dir saves/internlm-20b-chat/lora/internlm-20b-alpaca-zh-sample5000-bit4 \
    --overwrite_output_dir \
    --max_samples 5000 \
    --num_train_epochs 3.0 \
    --logging_steps 10 \
    --save_steps 800 \
    --val_size 0.1 \
    --evaluation_strategy no \
    --eval_steps 800 \
    --quantization_bit 4 \
    --plot_loss True \
    --load_best_model_at_end False
}

run_internlm_20b_deepspeed() {
    deepspeed -i localhost:4,5,6,7 --master_port=9902 src/train_bash.py \
    --deepspeed scripts/ds_config_zero3.json \
    --stage sft \
    --model_name_or_path /home/public/llm/internlm-20b-chat/ \
    --do_train True \
    --finetuning_type lora \
    --template intern \
    --lora_target q_proj,v_proj \
    --dataset_dir data \
    --dataset alpaca_zh \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --per_device_train_batch_size 12 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --output_dir saves/internlm-20b-chat/lora/internlm-20b-alpaca-zh-sample5000-deepspeed \
    --overwrite_output_dir \
    --max_samples 5000 \
    --num_train_epochs 3.0 \
    --logging_steps 10 \
    --save_steps 800 \
    --val_size 0.1 \
    --evaluation_strategy no \
    --eval_steps 800 \
    --plot_loss True \
    --fp16 True \
    --load_best_model_at_end False
}

run_predict_internlm_20b() {
    CUDA_VISIBLE_DEVICES=3 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/public/llm/internlm-20b-chat/ \
    --do_predict \
    --dataset_dir data \
    --dataset alpaca_zh \
    --template intern \
    --finetuning_type lora \
    --checkpoint_dir saves/internlm-20b-chat/lora/internlm-20b-alpaca-zh-sample5000-deepspeed \
    --output_dir saves/internlm-20b-chat/lora/internlm-20b-alpaca-zh-sample5000-deepspeed \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --quantization_bit 4 \
    --max_samples 500
}

# run_internlm_20b_qlora4
# run_internlm_20b_deepspeed
run_predict_internlm_20b
