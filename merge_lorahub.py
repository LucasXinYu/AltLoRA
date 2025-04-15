"""
Usage:
python merge_lora.py --base_model_path [BASE-MODEL-PATH] --lora_path [LORA-PATH]
"""
import argparse
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def merge_lora(base_model_name, lora_path):

    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)

    model = peft_model.merge_and_unload()
    model = model.to(torch.bfloat16)
    target_model_path = 'checkpoints/llama3_aslora_Adamw_altfalse_lr3e-05_a8_r8_s8_seed31' + "_full"
    model.save_pretrained(target_model_path)
    tokenizer.save_pretrained(target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--lora_path", type=str, required=True)

    args = parser.parse_args()

    merge_lora(args.base_model_path, args.lora_path)

# python merge_lorahub.py --base_model_path meta-llama/Meta-Llama-3-8B --lora_path lucas1026/llama3_aslora_Adamw_altfalse_lr3e-05_a8_r8_s8_seed31