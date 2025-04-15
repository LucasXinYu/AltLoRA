---
library_name: peft
license: llama3
base_model: meta-llama/Meta-Llama-3-8B
tags:
- generated_from_trainer
datasets:
- meta-math/MetaMathQA
model-index:
- name: 1e-4_adamwr
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# 1e-4_adamwr

This model is a fine-tuned version of [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) on the meta-math/MetaMathQA default dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.05
- training_steps: 1000

### Training results



### Framework versions

- PEFT 0.11.2.dev0
- Transformers 4.45.2
- Pytorch 2.6.0+cu124
- Datasets 2.19.0
- Tokenizers 0.20.3