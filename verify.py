import os
import random
from datasets import load_dataset, Dataset, load_from_disk
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, GenerationConfig, GemmaTokenizer, GemmaForCausalLM
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer,AutoModelForCausalLM, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, LogitsProcessorList, MinLengthLogitsProcessor, AutoModelForSequenceClassification
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType
import torch
import numpy as np
from argparse import ArgumentParser
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import normalize
from tqdm import tqdm
import torch.nn.functional as F
# from evalplus.data import get_mbpp_plus, write_jsonl
# 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
seed = 42

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

hftoken = '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'

train_dataset = load_dataset("TIGER-Lab/MathInstruct", 'default', split='train')
# eval_dataset = load_dataset("gsm8k", 'main', split='test')
train_dataset=train_dataset.rename_column('instruction', 'inputs')
train_dataset=train_dataset.rename_column('output', 'targets')

# syn_dataset = load_from_disk("datasyn/8b_mathins")
syn_dataset = train_dataset.select(range(500))
model_name = "meta-llama/Llama-3.2-3B"  # 请根据实际模型版本修改
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=torch.bfloat16, num_labels=2)
model = model.to(torch.device('cuda:0'))

meta_model_name = "/data/public_models/Meta-Llama-3-8B"
# "/data/public_models/Meta-Llama-3-8B"
metatokenizer = AutoTokenizer.from_pretrained(meta_model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(meta_model_name, torch_dtype=torch.float16, device_map="auto")
metatokenizer.pad_token_id = metatokenizer.eos_token_id
metamodel = PeftModel.from_pretrained(base_model, "/data/yujia_wang/translora/checkpoints_moe/Meta-Llama-3-8B/TIGER-Lab/MathInstruct/1e-6_noniid_disc0")
metamodel = metamodel.to(torch.device('cuda:0'))

ml = 512
yes_token = tokenizer.convert_tokens_to_ids("Yes")
no_token = tokenizer.convert_tokens_to_ids("No")
eos_token_id = torch.tensor(tokenizer.eos_token_id, device=model.device)  # 确保在 GPU
print(tokenizer.chat_template)  # 检查是否有 chat 模板

def classify_text_with_prob(text):
    prompt = f"""[INST] <<SYS>>\nAnswer with only 'Yes' or 'No'.\n<</SYS>>\n\n{text}\nIs this question from the math dataset? [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = metamodel(**inputs)
    
    logits = outputs.logits
    probs = F.softmax(logits, dim=-1)  # 计算 softmax 概率
    
    yes_prob = probs[0][1].item()  # "Yes" 的概率
    no_prob = probs[0][0].item()  # "No" 的概率
    
    prediction = True if yes_prob > no_prob else False
    # prediction = "Yes" if yes_prob > no_prob else "No"
    print(f'Yes: {yes_prob}, No: {no_prob}', flush=True)
    
    return prediction, {"Yes": yes_prob, "No": no_prob}

sum=0
for sample in syn_dataset:
    text = sample["inputs"]
    predicted_label, prob = classify_text_with_prob(text)  # 你的概率预测方法
    # print(f"Text: {text}")
    # print(f"Predicted Label: {predicted_label}")
    # print(f"Probabilities: {prob}")  # 打印 "Yes" / "No" 置信度
    if predicted_label == True:
        sum += 1
        print(text)
    print("="*50)
print(sum)
# def classify_text(text):
#     # 预处理输入
#     inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
#     # 让模型前向传播
#     outputs = model(**inputs)
    
#     # 取 logits，并计算概率
#     logits = outputs.logits
#     prediction = torch.argmax(logits, dim=-1).item()
    
#     # 转换为 "Yes" / "No"
#     return "Yes" if prediction == 1 else "No"

# test_text = "A boy finds 5 oranges. If he eats 2, how many are left?"
# print(classify_text(test_text))  # "Yes" 或 "No"
# def classify_text(text):
#     prompt = f"You must answer only with 'Yes' or 'No'. No explanation, no extra text.\n\n{text}\nIs this question from the gsm8k dataset? Answer with 'Yes' or 'No' only."

#     inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

#     out = model.generate(
#             **inputs,
#             max_new_tokens=3,  # 限制生成长度
#             do_sample=False,
#             repetition_penalty=1.5,  # 防止复述
#             top_k=2  # 只允许 "Yes" 或 "No"
#         )
#     print(tokenizer.decode(out[0]))  # 检查模型输出
#     output = tokenizer.decode(out[0], skip_special_tokens=True).strip()
#     return output

# def verify(inputstr):
    
#     prompt = f"[INST] <<SYS>>\nAnswer with only 'Yes' or 'No'.\n<</SYS>>\n\n{inputstr}\n Is the above question from the gsm8k dataset? [/INST]. "
#     # prompt = f"[INST] <<SYS>>\nAnswer in as few words as possible.\n<</SYS>>\n\n{inputstr}\n Is the above question from the gsm8k dataset? Assistant: [/INST]. "

#     # prompt = f"<start_of_turn>user\nAnswer in as few words as possible.\n\n{inputstr}\nIs the above question from the {taskname.replace('_', ' ')} dataset?<end_of_turn>\n<start_of_turn>model\n"

#     out = metamodel.generate(inputs=metatokenizer(prompt, return_tensors="pt").input_ids.to(model.device),do_sample=True, max_new_tokens=ml)
#     output = metatokenizer.decode(out[0])
#     # output = output[len(prompt)+6:]
#     # output = output[len(prompt)+5:]
#     # print(output[:3])
#     print(out.shape)
#     print(output)
#     if (output[:3] == 'Yes'):
#         return True
#     else:
#         return False
# train_dataset = load_dataset("gsm8k", 'main', split='train')
# train_dataset=train_dataset.rename_column('question', 'inputs')
# train_dataset=train_dataset.rename_column('answer', 'targets')

# def genoutput7b(inputt):
#     out = model.generate(inputs=tokenizer(inputt, return_tensors="pt").input_ids.to(model.device), do_sample=True, max_length=ml)
#     output = tokenizer.decode(out[0], do_sample=False)
#     # print(output,flush=True)
#     out_str = output[len(inputt)+4:]
#     return out_str

# temp_pro = 'Here are 10 examples:\n1. ' + train_dataset[0]['inputs'] + '\n2. ' + train_dataset[1]['inputs'] + '\n3. ' + train_dataset[2]['inputs'] + '\n4. ' + train_dataset[3]['inputs'] + '\n5. ' + train_dataset[4]['inputs'] + '\n6.'
# # print(temp_pro)
# for i in range(2):
#     generated = []
#     out = genoutput7b(temp_pro)
#     out = out.split('\n7.')[0]
#     # print(out)
#     a = verify(out)
#     print(a)
#     generated.append(out)
