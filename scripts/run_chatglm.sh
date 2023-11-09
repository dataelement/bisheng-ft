# CUDA_VISIBLE_DEVICES=0 python src/train_web.py

run_chatglm2_6b() {
    CUDA_VISIBLE_DEVICES=4 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/public/llm/chatglm2-6b \
    --do_train True \
    --finetuning_type lora \
    --template chatglm2 \
    --lora_target query_key_value \
    --dataset_dir data \
    --dataset alpaca_zh \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --output_dir saves/ChatGLM2-6B-Chat/lora/chatglm2-6b-alpaca-zh-sample5000 \
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

run_chatglm2_6b_qlora4() {
    CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/public/llm/chatglm2-6b \
    --do_train True \
    --finetuning_type lora \
    --template chatglm2 \
    --lora_target query_key_value \
    --dataset_dir data \
    --dataset alpaca_zh \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --output_dir saves/ChatGLM2-6B-Chat/lora/chatglm2-6b-alpaca-zh-sample5000-bit4 \
    --overwrite_output_dir \
    --max_samples 5000 \
    --num_train_epochs 3.0 \
    --logging_steps 10 \
    --save_steps 800 \
    --val_size 0.1 \
    --evaluation_strategy no \
    --eval_steps 800 \
    --plot_loss True \
    --quantization_bit 4 \
    --load_best_model_at_end False
}

run_chatglm2_6b_qlora8() {
    CUDA_VISIBLE_DEVICES=6 python src/train_bash.py \
        --stage sft \
        --model_name_or_path /home/public/llm/chatglm2-6b \
        --do_train True \
        --finetuning_type lora \
        --template chatglm2 \
        --lora_target query_key_value \
        --dataset_dir data \
        --dataset alpaca_zh \
        --cutoff_len 1024 \
        --learning_rate 5e-05 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --lr_scheduler_type cosine \
        --output_dir saves/ChatGLM2-6B-Chat/lora/chatglm2-6b-alpaca-zh-sample5000-bit8 \
        --overwrite_output_dir \
        --max_samples 5000 \
        --num_train_epochs 3.0 \
        --logging_steps 10 \
        --save_steps 800 \
        --val_size 0.1 \
        --evaluation_strategy no \
        --eval_steps 800 \
        --plot_loss True \
        --quantization_bit 8 \
        --load_best_model_at_end False
}

run_predict_chatglm2_6b() {
    CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/public/llm/chatglm2-6b \
    --do_predict \
    --dataset_dir data \
    --dataset alpaca_zh \
    --template chatglm2 \
    --finetuning_type lora \
    --checkpoint_dir saves/ChatGLM2-6B-Chat/lora/chatglm2-6b-alpaca-zh-sample5000-bit8 \
    --output_dir saves/ChatGLM2-6B-Chat/lora/chatglm2-6b-alpaca-zh-sample5000-bit8 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --quantization_bit 8 \
    --max_samples 500
}

run_chatglm2_6b
# run_chatglm2_6b_qlora4
# run_chatglm2_6b_qlora8
# run_predict_chatglm2_6b