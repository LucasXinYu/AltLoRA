#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.





"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
from peft.tuners.lora.model import MoELinear

import logging
import math
import os
import sys
from accelerate import Accelerator

#/root/autodl-tmp/aslora_new/peft
sys.path.insert(0, '/root/autodl-tmp/aslora_new/peft')
# pwd = '' # You should provice the work directory. 
# sys.path = [os.path.abspath(os.path.join(os.getcwd(), " "))] + sys.path

from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
from peft.utils import _get_submodules

import datasets
#from datasets import load_dataset,load_from_disk, load_metric
from datasets import load_dataset, load_from_disk
import evaluate
from peft import LoraConfig, TaskType,PeftType, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, PeftModel
from peft import LoraConfig, get_peft_model, PeftType
import torch.nn as nn
import torch
import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft.tuners.lora.layer import Linear as LoraLinear

from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


from utils import get_training_args, cosine_learning_rate
import copy
from tqdm import tqdm
import random
from trl import SFTTrainer
import torch.nn.utils.prune as prune
import torch.nn as nn
import torch
import numpy as np
import pickle
import re
from logtrainer import LogTrainer


from huggingface_hub import login, HfApi, create_repo
import wandb


# Will error if the minimal version of moe_transformers is not installed. Remove at your own risks.
# check_min_version("4.26.0.dev0")
sys.setrecursionlimit(30000)

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

LLAMA3_CHAT_TEMPLATE = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
LLAMA32_CHAT_TEMPLATE = "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n    {%- else %}\n        {%- set date_string = \"26 Jul 2024\" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n        {{- '\"parameters\": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- \"}\" }}\n        {{- \"<|eot_id|>\" }}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n",






@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    process_dim: int = field(
        default=128,
    )
    


    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    max_gate_samples: Optional[int] = field(default=None, metadata={"help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=10,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class FedArguments:
    fed_alg: Optional[str] = field(default="fedavg", metadata={"help": "the algorithm to use"})
    num_rounds: Optional[int] = field(default=20, metadata={"help": "the number of rounds"})
    num_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients"})
    sample_clients: Optional[int] = field(default=2, metadata={"help": "the number of clients to sample"})
    split_strategy: Optional[str] = field(default="iid", metadata={"help": "the split strategy"})
    prox_mu: Optional[float] = field(default=0.01, metadata={"help": "the mu parameter of FedProx"})
    fedopt_tau: Optional[float] = field(default=1e-3, metadata={"help": "the tau parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_eta: Optional[float] = field(default=1.0, metadata={"help": "the global learning rate parameter of FedAdagrad, FedYogi and FedAdam"})
    fedopt_beta1: Optional[float] = field(default=0.9, metadata={"help": "the beta1 parameter of FedYogi and FedAdam"})
    fedopt_beta2: Optional[float] = field(default=0.99, metadata={"help": "the beta2 parameter of FedYogi and FedAdam"})
    save_model_freq: Optional[int] = field(default=50, metadata={"help": "the frequency to save the model. 50 means save every 50 rounds"})
    optim_notes: Optional[str] = field(default="optim_notes", metadata={"help": "the optim_notes of the experiment"})

#####################################################################
#####################################################################
def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, FedArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, fed_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, fed_args = parser.parse_args_into_dataclasses()




    f_read_token = "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"
    hf_write_token = "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"
    hf_repo_id = "lucas1026/aslora"
    
    # 1. 登录 Hugging Face Hub 读取权限（用于下载模型、数据集等）
    if f_read_token:
        from huggingface_hub import login
        login(token=f_read_token)
        os.environ["HUGGINGFACE_HUB_TOKEN"] = f_read_token  # 有些库会用这个环境变量

    # 2. 登录 WandB
    wandb.login(key="4103935e0eb3f62d9b7be7d002fb27a4180e8399")
    accelerator = Accelerator()
    
    lora_alpha = 16
    lora_rank = 8
    batchsize = 1
    lora_alt = False 


    print(fed_args.optim_notes)
    config = dict(
        model="meta-llama/Meta-Llama-3-8B",
        d="meta_math",
        a=lora_alpha,
        r=lora_rank,
        s=batchsize,#real batch size?
        sd=42,
        optim_name=fed_args.optim_notes,
        alt=lora_alt,
        lr=training_args.learning_rate,
    )
    print(config)
    
    
    wandb_name = "_".join([f"{k}={v}" for k, v in config.items()])
    if accelerator.is_local_main_process:
        wandb.init(
            name=wandb_name,
            mode="online",
            group="train",
            project="ALoRA",
        )

    
   
    
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

   
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print('fdasfdasfdafaffsadfdasfdsafasdfdasfdsafasdf')
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
           data_args.dataset_name,
           data_args.dataset_config_name,
           cache_dir=model_args.cache_dir,
           use_auth_token=True if model_args.use_auth_token else None,
        )        

        # raw_datasets = load_from_disk('/aifs4su/data/zhaohao/wikitext')
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
            raw_datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
            )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    ## Load pretrained model and tokenizer ##
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    setattr(config, 'seed', training_args.seed)

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if "llama-3.2-1b-instruct" in model_args.model_name_or_path.lower():
        pass 
    elif "llama-3.2-" in model_args.model_name_or_path.lower() and "instruct" not in model_args.model_name_or_path.lower():
        tokenizer.chat_template = LLAMA32_CHAT_TEMPLATE
    elif "llama-3-" in model_args.model_name_or_path.lower() and "instruct" not in model_args.model_name_or_path.lower():
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE

    if model_args.model_name_or_path:
        # Set torch dtype and attention implementation
        if torch.cuda.get_device_capability()[0] >= 8:
            torch_dtype = torch.bfloat16
            attn_implementation = "flash_attention_2"
        else:
            torch_dtype = torch.float16
            attn_implementation = "eager"
        print(attn_implementation)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            torch_dtype=torch_dtype,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model.config.attn_implementation = attn_implementation
        
                

    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    print(f"Model dtype: {next(model.parameters()).dtype}")

    num_layers = sum(1 for _ in model.parameters())
    print(f"Number of layers (parameter sets) in the model: {num_layers}")

    print(f'max_gate_samples is {data_args.max_gate_samples}')
    #lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, task_type=TaskType.CAUSAL_LM, lora_dropout=0.05,
    #                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],) ## further to revise
    
    lora_config = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                task_type=TaskType.CAUSAL_LM,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_nums=4,                  # ✅ 
                blc_alpha=0.1,                # ✅
                blc_weight=1.0,               # ✅
                peft_type=PeftType.MOELORA,   # ✅ PeftType.LORA
            )
    
       
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.print_trainable_parameters()

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
        print(column_names)
        #column_names = raw_datasets["train"].column_names
    else:
        column_names = list(raw_datasets["validation"].features)
        #column_names = raw_datasets["validation"].column_names
    # text_column_name = "input" if "input" in column_names else column_names[-1]
    # print(text_column_name)
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("moe_transformers.tokenization_utils_base")
    
    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            if data_args.dataset_name == "meta-math/MetaMathQA":
                prompts = [
                    f"Question: {query} </s> Answer: "
                    for query in examples["query"]
                ]
                outputs = examples["response"]
                full_texts = [prompt + output for prompt, output in zip(prompts, outputs)]
            
            tokenized_full = tokenizer(full_texts)
            tokenized_prompts = tokenizer(prompts)
            labels = [ids.copy() for ids in tokenized_full["input_ids"]]
            for i, prompt_ids in enumerate(tokenized_prompts["input_ids"]):
                prompt_length = len(prompt_ids)
                labels[i][:prompt_length] = [-100] * prompt_length
            tokenized_full["labels"] = labels
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return tokenized_full
    
    #print(len(raw_datasets['train']))
    #exit()

    # Set the FL dataset
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(raw_datasets['train']), data_args.max_train_samples)
        raw_datasets['train'] = raw_datasets['train'].select(range(max_train_samples))

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    #print(len(lm_datasets['train']))
    #exit()

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def extract_option(text):
            pattern = r'(?:Answer\s*(?:is|:)?\s*|Ans\s*(?:is|:)?\s*|选项\s*)?\b([A-E])\b'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            return None

        def compute_metrics(eval_preds):
            preds, labels = eval_preds

            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)

            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            correct = 0
            extracted_labels = [extract_option(label) for label in decoded_labels]
            valid_idx = [i for i, label in enumerate(extracted_labels) if label is not None]
            total = len(valid_idx)
            extracted_preds = []
            extracted_labels = []

            for pred, label in zip(decoded_preds, decoded_labels):
                pred_option = extract_option(pred)
                label_option = extract_option(label)
                extracted_preds.append(pred_option)
                extracted_labels.append(label_option)
                if pred_option is not None and pred_option == label_option:
                    correct += 1

            accuracy = correct / total if total > 0 else 0.0

            invalid_preds = sum(1 for opt in extracted_preds if opt is None)
            invalid_labels = sum(1 for opt in extracted_labels if opt is None)

            metrics = {
                "accuracy": accuracy,
                "total": total,
                "invalid_preds": invalid_preds,
                "invalid_labels": invalid_labels,
            }
            return metrics

    train_loss = []
    val_loss = []

    for round in tqdm(range(fed_args.num_rounds)):
        training_loss = []
        training_args_new = get_training_args(training_args, training_args.learning_rate)
        trainer = LogTrainer(
                model=model,
                args=training_args_new,
                train_dataset=lm_datasets["train"] if training_args.do_train else None,
                # train_dataset=sub_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                tokenizer=tokenizer,
                optim_name=fed_args.optim_notes,
                # Data collator will default to DataCollatorWithPadding, so we change it.
                data_collator=default_data_collator,
                compute_metrics=compute_metrics if training_args.do_eval else None,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics
                if training_args.do_eval 
                else None,
                rank = lora_rank,
                alt = lora_alt,
                
                
            )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            print('checkpoint')
            print(checkpoint)
            train_result = trainer.train()#trainer.train(resume_from_checkpoint=checkpoint)
            training_loss.append(train_result.training_loss)

            metrics = train_result.metrics

            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(lm_datasets["train"])
            )
            metrics["train_samples"] = min(max_train_samples, len(lm_datasets["train"]))
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

        # for ld in local_dict_list:
        #     last_key = list(ld.keys())[-1]  
        #     print(last_key, ld[last_key])   
        # breakpoint()
        train_loss.append(np.mean(training_loss))
        # model.load_state_dict(global_dict) 
        # if fed_args.num_rounds>200:
        if data_args.dataset_name == "meta-math/MetaMathQA":
            pass
        else: 
            if training_args.do_eval:
                logger.info("*** Evaluate ***")
                trainer.model = model

                metrics = trainer.evaluate()
                max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
                if data_args.dataset_name == "TIGER-Lab/MathInstruct":
                    metrics["eval_samples"] = 1000
                else:
                    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
                try:
                    perplexity = math.exp(metrics["eval_loss"])
                except OverflowError:
                    perplexity = float("inf")
                metrics["perplexity"] = perplexity

                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

                val_loss.append(metrics["eval_loss"])
        model.to(training_args.device)

        kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name
        
    config_suffix = f"lr-{training_args.learning_rate}_r-{lora_rank}_alpha-{lora_alpha}"
    hf_repo_id = f"{hf_repo_id}_{config_suffix}"
    
    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    
    
    print("\n========== MODEL STRUCTURE ==========")
    print(model)

    print("\n========== TRAINABLE PARAMETERS ==========")
    trainable_params = []
    total_params = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params.append((name, param.shape, param.numel()))
            trainable_count += param.numel()

    for name, shape, num in trainable_params[:50]:  
        print(f"{name}: shape={shape}, numel={num}")

    print(f"\nTotal trainable parameters: {trainable_count:,} / {total_params:,} ({100 * trainable_count / total_params:.2f}%)")
    
    
    ###############################################################################################
    #####################################################################################
    


    def convert_moe_lora_to_strongest_expert_adapter(model):
        """
        Convert each MoELinear layer in a PEFT model into a LoraLinear layer,
        by selecting the expert with the highest std in B weight.
        """
        for name, module in model.named_modules():
            if type(module).__name__ == "MoELinear":
                print(f"[Converting MoE → LoRA @ strongest expert] {name}")

                # Step 1: 找到 std 最大的 expert
                best_idx = max(
                    range(len(module.lora_Bs)),
                    key=lambda i: module.lora_Bs[i].weight.data.std().item()
                )

                A_best = module.lora_As[best_idx].weight.data
                B_best = module.lora_Bs[best_idx].weight.data

                print(f"  → Selected Expert {best_idx} | B std = {B_best.std().item():.6f}")

                # Step 2: 构建标准 LoRA Linear 层
                merged = LoraLinear(
                    base_layer=module.base_layer,
                    adapter_name="default",
                    r=module.r,
                    lora_alpha=module.lora_alpha,
                    lora_dropout=0.0,
                )

                # Step 3: 填入选中的 expert 的权重
                merged.lora_A["default"].weight.data = A_best.clone()
                merged.lora_B["default"].weight.data = B_best.clone()

                # Step 4: 替换原始模块
                parent = model
                subnames = name.split(".")
                for sub in subnames[:-1]:
                    parent = getattr(parent, sub)
                setattr(parent, subnames[-1], merged)

        return model
    
  

    def convert_moelora_model_to_lora_model_strongest_only(peft_model):
        """
        Convert a MoELoraModel to standard LoraModel using only strongest expert per layer.
        """
        from peft.tuners.lora import LoraModel

        # 1. 选取每层最强 expert → LoraLinear
        peft_model = convert_moe_lora_to_strongest_expert_adapter(peft_model)

        # 2. 构建标准 LoraModel（非 MoE）
        base = peft_model.base_model
        model = base.model
        lora_config = peft_model.peft_config["default"]

        new_base_model = LoraModel(model, lora_config, adapter_name="default")
        peft_model.base_model = new_base_model

        return peft_model


    model = convert_moelora_model_to_lora_model_strongest_only(model)
    print(type(model.base_model))  # 应为 peft.tuners.lora.LoraModel
    print(model.base_model.model.model.layers[0].self_attn.q_proj)  # 应为 peft.tuners.lora.layer.LoraLinear




    ##########################################################################################
    ##########################################################################################
    ##########################################################################################
    
    
    print("\n========== MODEL STRUCTURE ==========")
    print(model)

    print("\n========== TRAINABLE PARAMETERS ==========")
    trainable_params = []
    total_params = 0
    trainable_count = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params.append((name, param.shape, param.numel()))
            trainable_count += param.numel()

    for name, shape, num in trainable_params[:50]:  
        print(f"{name}: shape={shape}, numel={num}")

    print(f"\nTotal trainable parameters: {trainable_count:,} / {total_params:,} ({100 * trainable_count / total_params:.2f}%)")
    
    

    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            print(f"\n[LoraLinear Layer] {name}")
            for adapter_name in module.lora_A.keys():
                A = module.lora_A[adapter_name].weight
                B = module.lora_B[adapter_name].weight
                print(f"  Adapter: {adapter_name}")
                print(f"    A: shape={A.shape}, mean={A.mean().item():.6f}, std={A.std().item():.6f}")
                print(f"    B: shape={B.shape}, mean={B.mean().item():.6f}, std={B.std().item():.6f}")



    model.push_to_hub(hf_repo_id, use_auth_token=hf_write_token)

    
    trainer.create_model_card(**kwargs)
    # Restore k,v cache for fast inference
    trainer.model.config.use_cache = True
    # lora_config.save_pretrained(final_training_args.output_dir)
    trainer.model.save_pretrained(training_args_new.output_dir)
    tokenizer.save_pretrained(training_args_new.output_dir)
    trainer.save_model(training_args_new.output_dir)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

