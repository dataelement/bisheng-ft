bisheng_ft train \
--model_name_or_path /home/public/llm/Qwen-7B-Chat \
--dataset /home/gulixin/workspace/llm/bisheng-ft/data/alpaca_data_zh_51k.json,/home/gulixin/workspace/llm/bisheng-ft/data/alpaca_data_en_52k.json \
--output_dir saves/cli_test/lora/qwen-7b-adgen-val-sample5000-bit4 \
--finetuning_type lora \
--per_device_train_batch_size 4 \
--max_samples 1000 \
--gpus 0 \