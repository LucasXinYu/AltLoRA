

ACCELERATE_LOG_LEVEL=info accelerate launch --num_processes=4 --main_process_port 29504 --config_file accelerate_configs/deepspeed_zero3.yaml training_scripts/run_dpo.sh


python3 merge_lora.py --base_model_path "/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B" --lora_path "/data/yujia_wang/fedmoe_llama/moe/checkpoints_moe/Meta-Llama-3-8B/TIGER-Lab/MathInstruct/5e-5_noniid_r"

alpaca_eval evaluate_from_model --model_configs 'my_sft_fl' --chunksize 1 --annotators_config gpt-4o-mini

CUDA_VISIBLE_DEVICES=0 python3 run_sft.py training_configs/llama-3-8b-base-sft.yaml > log.txt

CUDA_VISIBLE_DEVICES=0 python3 run_sft_iid.py training_configs/llama-3-8b-base-sft_iid.yaml > log2.txt

CUDA_VISIBLE_DEVICES=0 python3 run_sft_5c.py training_configs/llama-3-8b-base-sft_5c.yaml > log3.txt


lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-3B \
    --tasks pubmedqa \
    --device cuda:0 \
    --output_path ./eval_out/pubmedqa

lm_eval --model hf \
    --model_args pretrained=AdaptLLM/medicine-chat \
    --tasks medqa_4options \
    --device cuda:0 \
    --output_path ./eval_out/medqa

lm_eval --model hf \
    --model_args pretrained=/data/yujia_wang/fedmoe_llama/moe/checkpoints_moe/Llama-3.2-3B/HPAI-BSC/medqa-cot/3e-6_bnoniid_bcd \
    --tasks medmcqa \
    --device cuda:0 \
    --output_path ./eval_out/medmcqa

lm_eval --model hf \
    --model_args pretrained=/data/yujia_wang/fedmoe_llama/moe/checkpoints_moe/Meta-Llama-3-8B/medalpaca/medical_meadow_medical_flashcards/1e-5_bnoniid_bcd \
    --tasks mmlu_continuation_professional_medicine \
    --device cuda:0 \
    --output_path ./eval_out/mmlu_pm

python3 merge_lora.py --base_model_path "/data/public_models/Meta-Llama-3-8B" --lora_path "/data/yujia_wang/translora/checkpoints_moe/Meta-Llama-3-8B/openai/gsm8k/1e-7_iid_lora"

python3 merge_lora.py --base_model_path "meta-llama/Llama-3.2-3B" --lora_path "/data/yujia_wang/translora/checkpoints_moe/Llama-3.2-3B/openai/gsm8k/3e-6_iid_lora"


lm_eval --model hf \
    --model_args pretrained=/data/yujia_wang/translora/checkpoints_moe/Llama-3.2-3B/openai/gsm8k/3e-6_iid_lora_full \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 8 \
    --device cuda:0 \
    --output_path ./eval_out/gsm8k_cot

lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct \
    --tasks boolq \
    --batch_size 8 \
    --device cuda:0 \
    --output_path ./eval_out/boolq



lm_eval --model hf \
    --model_args pretrained=meta-llama/Llama-3.2-3B \
    --tasks bbh_fewshot_boolean_expressions \
    --batch_size auto \
    --device cuda:0 \
    --output_path ./eval_out/bbh

CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=/data/yujia_wang/fedmoe_llama/moe/checkpoints_moe/Llama-3.2-1B/vicgalle/alpaca-gpt4/1e-6_iid_layer \
    --tasks leaderboard_math_hard \
    --batch_size auto \
    --num_fewshot 4 \
    --output_path ./eval_out/llama32_math

CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=/data/yujia_wang/fedmoe_llama/moe/checkpoints_moe/Llama-3.2-3B/vicgalle/alpaca-gpt4/1e-6_bnoniid_layer \
    --tasks arc_challenge \
    --batch_size auto \
    --num_fewshot 25 \
    --output_path ./eval_out/arc

CUDA_VISIBLE_DEVICES=0 lm_eval --model hf \
    --model_args pretrained=/data/public_models/Meta-Llama-3-8B \
    --tasks mmlu \
    --batch_size auto \
    --num_fewshot 5 \
    --output_path ./eval_out/mmlu





winogrande
cp -r /data/yujia_wang/OpenFedLLM/lm-evaluation-harness /data/yujia_wang/fedmoe_llama/moe


python3 fingpt_eval.py --base_model_path "/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B" --lora_path "/data/yujia_wang/fedmoe_llama/moe/checkpoints_moe/Meta-Llama-3-8B/FinGPT/fingpt-sentiment-train/5e-5_1_noniid_sv" > log_n.txt

python3 fingpt_eval.py --base_model_path "/data/public_models/huggingface/meta-llama/Meta-Llama-3-8B" --lora_path "/data/yujia_wang/fedmoe_llama/moe/checkpoints_moe/Meta-Llama-3-8B/zeroshot/twitter-financial-news-sentiment/1e-4_1_iid_sv" > log.txt


python detail_memory.py /data/public_models/Llama-2-7b-chat-hf 1024  0 0  128 # base
python detail_memory.py /data/public_models/Llama-2-7b-chat-hf 1024  0 1  128 # Lora
python ./examples/detail_memory.py meta-llama/Llama-2-7b-hf 1024  0 1  256 # Lora
python detail_memory.py /data/public_models/Meta-Llama-3-8B 1024  0 1  4 # Lora


lm_eval --model hf \
    --model_args pretrained=checkpoints_moe/baseline_standardlora_alternative_aslora_Adamw_alttrue_lr1e-05_a8_r8_s128_seed31_full \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 8 \
    --device cuda:2 \
    --output_path ./eval_out/gsm8k_cot

bash run_main_math2.sh > logm2.txt
python3 merge_lora.py --base_model_path "/data/public_models/Meta-Llama-3-8B" --lora_path "/data/yujia_wang/translora/checkpoints_moe/Meta-Llama-3-8B/openai/gsm8k/3e-6_bnoniid_lora"
lm_eval --model hf \
    --model_args pretrained=checkpoints_moe/Meta-Llama-3-8B/medalpaca/medical_meadow_medqa/3e-6_iid_lora_full \
    --tasks medqa_4options \
    --batch_size auto \
    --device cuda:0 \
    --output_path ./eval_out/medqa_4options
