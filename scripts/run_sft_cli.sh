bisheng_ft train \
--model_name_or_path /home/public/llm/Qwen-7B-Chat \
--dataset alpaca_zh 
--output_dir saves/cli_test/lora/qwen-7b-adgen-val-sample5000-bit4
--finetuning_type lora 
--output_dir saves/Qwen-7B-Chat/lora/qwen-7b-adgen-val-sample5000-bit4 --per_device_eval_batch_size 1 