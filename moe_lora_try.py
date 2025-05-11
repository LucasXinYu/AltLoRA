import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

model_id = "meta-llama/Meta-Llama-3-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 配置 MoE-LoRA adapter
peft_config = LoraConfig(
    peft_type="MOELORA",
    task_type="CAUSAL_LM",
    r=4,
    lora_alpha=32,
    lora_nums=8,
    blc_alpha=0.5,
    blc_weight=0.1,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# 注入 adapter
model = get_peft_model(model, peft_config)

# 显示 LoRA 参数数量
model.print_trainable_parameters()

# Dummy inference
input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    outputs = model(input_ids=input_ids)

print("✅ MoE-LoRA injection and forward pass succeeded.")
