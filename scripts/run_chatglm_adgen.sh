LR=2e-2
# LR=5e-5

run_chatglm2_6b_qlora4() {
    CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/public/llm/chatglm2-6b \
    --do_train True \
    --finetuning_type lora \
    --template chatglm2 \
    --lora_target query_key_value \
    --dataset_dir data \
    --dataset adgen_train \
    --cutoff_len 1024 \
    --learning_rate $LR \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --output_dir saves/ChatGLM2-6B-Chat/lora/chatglm2-6b-adgen-bit4-$LR \
    --overwrite_output_dir \
    --num_train_epochs 0.5 \
    --logging_steps 10 \
    --save_steps 800 \
    --val_size 0.0 \
    --evaluation_strategy no \
    --eval_steps 800 \
    --plot_loss True \
    --quantization_bit 4 \
    --load_best_model_at_end False
}

run_predict_chatglm2_6b() {
    CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/public/llm/chatglm2-6b \
    --do_predict \
    --dataset_dir data \
    --dataset adgen_val \
    --template chatglm2 \
    --finetuning_type lora \
    --checkpoint_dir saves/ChatGLM2-6B-Chat/lora/chatglm2-6b-adgen-bit4 \
    --output_dir saves/ChatGLM2-6B-Chat/lora/chatglm2-6b-adgen-bit4 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --quantization_bit 4
}

run_chatglm2_6b_qlora4
# run_predict_chatglm2_6b