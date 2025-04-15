from datasets import load_dataset
import re

from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Subset
import random
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd

from transformers import HfArgumentParser, TrainingArguments, BitsAndBytesConfig, set_seed
import math
import json
from datasets import concatenate_datasets
import copy 
from collections import defaultdict


def get_training_args(script_args, new_lr):
    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=new_lr,
        # logging_steps=script_args.logging_steps,
        logging_steps=5,
        num_train_epochs=script_args.num_train_epochs,
        max_steps=script_args.max_steps,
        save_steps=script_args.save_steps,
        save_strategy=script_args.save_strategy,
        save_total_limit=script_args.save_total_limit,
        push_to_hub=script_args.push_to_hub,
        hub_model_id=script_args.hub_model_id,
        gradient_checkpointing=script_args.gradient_checkpointing,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=0.05,
        # warmup_ratio=script_args.warmup_ratio,
        bf16=True,
        report_to="wandb",
        # report_to=script_args.report_to,
    )
    return training_args

def cosine_learning_rate(total_rounds, initial_lr=0.001):
    
    optimizer_setup = torch.optim.SGD([torch.randn(1, requires_grad=True)], lr=initial_lr)

    # lr schedule
    from lr_schedulers import CosineLR
    scheduler_setup = CosineLR(optimizer_setup, warmup_length=5, end_epoch=total_rounds*1.2)
    lrs = []

    for epoch in range(total_rounds):
        scheduler_setup.step()
        lrs.append(scheduler_setup.get_last_lr()[0])
    return lrs

def get_dataset_this_round(dataset, round, script_args, num_layers):
    print(int(len(dataset) /num_layers))
    num2sample = script_args.per_device_train_batch_size * script_args.gradient_accumulation_steps * int(len(dataset) /num_layers)
        # num2sample = script_args.per_device_train_batch_size * script_args.gradient_accumulation_steps * int(len(dataset) /16)
    num2sample = min(max(1,num2sample), len(dataset))
    random.seed(round)
    random_idx = random.sample(range(0, len(dataset)), num2sample)
    dataset_this_round = dataset.select(random_idx)

    return dataset_this_round

