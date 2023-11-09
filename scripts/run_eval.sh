CUDA_VISIBLE_DEVICES=1 python src/evaluate.py \
    --model_name_or_path /home/public/llm/Qwen-7B-Chat \
    --finetuning_type full \
    --template qwen \
    --task mmlu \
    --split validation \
    --lang zh \
    --n_shot 5 \
    --batch_size 1


CUDA_VISIBLE_DEVICES=4 python src/evaluate.py \
    --model_name_or_path /home/public/llm/Qwen-7B-Chat \
    --finetuning_type lora \
    --checkpoint_dir saves/Qwen-7B-Chat/lora/qwen-7b-alpaca-zh-sample5000 \
    --template qwen \
    --task mmlu \
    --split validation \
    --lang zh \
    --n_shot 5 \
    --batch_size 1


CUDA_VISIBLE_DEVICES=5 python src/evaluate.py \
    --model_name_or_path saves/Qwen-7B-Chat/full/qwen-7b-alpaca-zh-sample5000-deepspeed-full \
    --finetuning_type full \
    --template qwen \
    --task mmlu \
    --split validation \
    --lang zh \
    --n_shot 5 \
    --batch_size 1


CUDA_VISIBLE_DEVICES=6 python src/evaluate.py \
    --model_name_or_path saves/Qwen-7B-Chat/freeze/qwen-7b-alpaca-zh-sample5000-deepspeed-freeze \
    --finetuning_type full \
    --template qwen \
    --task mmlu \
    --split validation \
    --lang zh \
    --n_shot 5 \
    --batch_size 1