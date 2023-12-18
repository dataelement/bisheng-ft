run_qwen_deepspeed_full() {
    deepspeed -i localhost:1,3,4,5 --master_port=9902 src/train_bash.py \
    --deepspeed scripts/ds_config_zero3.json \
    --stage sft \
    --model_name_or_path /home/public/llm/Qwen-1_8B-Chat \
    --do_train True \
    --finetuning_type full \
    --template qwen \
    --dataset /home/gulixin/workspace/llm/projects/yuxin/workspace/train_samples.json \
    --cutoff_len 8192 \
    --learning_rate 5e-05 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --output_dir saves/Qwen-1_8B-Chat/full/qwen-1_8b-yuxin-deepspeed-full-8192 \
    --overwrite_output_dir \
    --num_train_epochs 3.0 \
    --logging_steps 10 \
    --save_strategy epoch \
    --evaluation_strategy epoch \
    --val_size 0.1 \
    --plot_loss True \
    --fp16 True \
    --metric_for_best_model eval_loss \
    --load_best_model_at_end True
}

run_predict_qwen_full() {
    CUDA_VISIBLE_DEVICES=5 python src/train_bash.py \
    --stage sft \
    --model_name_or_path saves/Qwen-1_8B-Chat/full/qwen-1_8b-yuxin-deepspeed-full \
    --do_predict \
    --dataset /home/gulixin/workspace/llm/projects/yuxin/workspace/test_samples.json \
    --cutoff_len 8192 \
    --template qwen \
    --finetuning_type full \
    --output_dir saves/Qwen-1_8B-Chat/full/qwen-1_8b-yuxin-deepspeed-full-8192 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --fp16 True
}

run_qwen_deepspeed_full
# run_predict_qwen_full