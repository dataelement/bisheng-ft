run_qwen_7b_qlora4() {
    CUDA_VISIBLE_DEVICES=5 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/public/llm/Qwen-7B-Chat \
    --do_train True \
    --finetuning_type lora \
    --template qwen \
    --lora_target c_attn \
    --dataset_dir data \
    --dataset alpaca_zh \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --output_dir saves/Qwen-7B-Chat/lora/qwen-7b-alpaca-zh-sample5000-bit4 \
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

run_qwen_14b_qlora4() {
    CUDA_VISIBLE_DEVICES=1 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/public/llm/Qwen-14B-Chat \
    --do_train True \
    --finetuning_type lora \
    --template qwen \
    --lora_target c_attn \
    --dataset_dir data \
    --dataset alpaca_zh \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --output_dir saves/Qwen-14B-Chat/lora/qwen-14b-alpaca-zh-sample5000-bit4 \
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

run_qwen_deepspeed_lora() {
    deepspeed -i localhost:4,5,6,7 --master_port=9902 src/train_bash.py \
    --deepspeed scripts/ds_config_zero3.json \
    --stage sft \
    --model_name_or_path /home/public/llm/Qwen-7B-Chat \
    --do_train True \
    --finetuning_type lora \
    --template qwen \
    --lora_target c_attn \
    --dataset_dir data \
    --dataset alpaca_zh \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --output_dir saves/Qwen-7B-Chat/lora/qwen-7b-alpaca-zh-sample5000-deepspeed \
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

run_qwen_deepspeed_full() {
    deepspeed -i localhost:4 --master_port=9902 src/train_bash.py \
    --deepspeed scripts/ds_config_zero3.json \
    --stage sft \
    --model_name_or_path /home/public/llm/Qwen-7B-Chat \
    --do_train True \
    --finetuning_type full \
    --template qwen \
    --dataset_dir data \
    --dataset alpaca_zh \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --output_dir saves/Qwen-7B-Chat/full/qwen-7b-alpaca-zh-sample5000-deepspeed-full \
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

run_qwen_deepspeed_freeze() {
    deepspeed -i localhost:4,5,6,7 --master_port=9902 src/train_bash.py \
    --deepspeed scripts/ds_config_zero2.json \
    --stage sft \
    --model_name_or_path /home/public/llm/Qwen-7B-Chat \
    --do_train True \
    --finetuning_type freeze \
    --template qwen \
    --dataset_dir data \
    --dataset alpaca_zh \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --output_dir saves/Qwen-7B-Chat/freeze/qwen-7b-alpaca-zh-sample5000-deepspeed-freeze \
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

run_predict_qwen_7b_lora() {
    CUDA_VISIBLE_DEVICES=3 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/public/llm/Qwen-7B-Chat \
    --do_predict \
    --dataset_dir data \
    --dataset alpaca_zh \
    --template qwen \
    --finetuning_type lora \
    --checkpoint_dir saves/Qwen-7B-Chat/lora/qwen-7b-alpaca-zh-sample5000-bit4 \
    --output_dir saves/Qwen-7B-Chat/lora/qwen-7b-alpaca-zh-sample5000-bit4 \
    --per_device_eval_batch_size 1 \
    --quantization_bit 4 \
    --predict_with_generate \
    --max_samples 500
}

run_predict_qwen_14b_lora() {
    CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/public/llm/Qwen-14B-Chat \
    --do_predict \
    --dataset_dir data \
    --dataset alpaca_zh \
    --template qwen \
    --finetuning_type lora \
    --checkpoint_dir saves/Qwen-14B-Chat/lora/qwen-14b-alpaca-zh-sample5000-deepspeed \
    --output_dir saves/Qwen-14B-Chat/lora/qwen-14b-alpaca-zh-sample5000-deepspeed \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --quantization_bit 4 \
    --max_samples 500
}

run_predict_qwen_7b_full() {
    CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --stage sft \
    --model_name_or_path saves/Qwen-7B-Chat/freeze/qwen-7b-alpaca-zh-sample5000-deepspeed-freeze \
    --do_predict \
    --dataset_dir data \
    --dataset alpaca_zh \
    --template qwen \
    --finetuning_type full \
    --output_dir saves/Qwen-7B-Chat/freeze/alpaca_zh \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --max_samples 500
}

run_predict_qwen_7b_lora_adgen() {
    CUDA_VISIBLE_DEVICES=3 python src/train_bash.py \
    --stage sft \
    --model_name_or_path /home/public/llm/Qwen-7B-Chat \
    --do_predict \
    --dataset_dir data \
    --dataset adgen_val \
    --template qwen \
    --finetuning_type lora \
    --checkpoint_dir saves/Qwen-7B-Chat/lora/qwen-7b-alpaca-zh-sample5000 \
    --output_dir saves/Qwen-7B-Chat/lora/adgen_val \
    --per_device_eval_batch_size 1 \
    --predict_with_generate 
}

run_predict_qwen_7b_full_adgen() {
    CUDA_VISIBLE_DEVICES=6 python src/train_bash.py \
    --stage sft \
    --model_name_or_path saves/Qwen-7B-Chat/freeze/qwen-7b-alpaca-zh-sample5000-deepspeed-freeze \
    --do_predict \
    --dataset_dir data \
    --dataset adgen_val \
    --template qwen \
    --finetuning_type full \
    --output_dir saves/Qwen-7B-Chat/freeze/adgen_val \
    --per_device_eval_batch_size 1 \
    --predict_with_generate
}

# run_qwen_7b_qlora4
# run_qwen_14b_qlora4
# run_qwen_deepspeed_lora
# run_qwen_deepspeed_full
# run_qwen_deepspeed_freeze
# run_predict_qwen_7b_lora
# run_predict_qwen_14b_lora
# run_predict_qwen_7b_full
# run_predict_qwen_7b_lora_adgen
# run_predict_qwen_7b_full_adgen