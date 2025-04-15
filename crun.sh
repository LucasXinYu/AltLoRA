
python3 merge_lora.py --base_model_path "meta-llama/Meta-Llama-3-8B" --lora_path "checkpoints/Meta-Llama-3-8B/meta-math/MetaMathQA/1e-4_adamwr"
lm_eval --model hf \
    --model_args pretrained=checkpoints/Meta-Llama-3-8B/meta-math/MetaMathQA/1e-4_adamwr_full \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 8 \
    --device cuda:3 \
    --output_path ./eval_out/gsm8k_cot
rm -r checkpoints/Meta-Llama-3-8B/meta-math/MetaMathQA/1e-4_adamwr_full
