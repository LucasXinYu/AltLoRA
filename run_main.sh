#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0

optim_notes=altlora # this indicates which optim we use
split_strategy=iid
num_rounds=1
num_clients=1
sample_clients=1
block_size=512
local_rank=8
fp16=True
lr_scheduler_type="cosine"

max_gate_samples=50
max_train_samples=100000
do_train=True

# ===[ MODIFY ]===
model_name_or_path="meta-llama/Meta-Llama-3-8B"
dataset_name=meta-math/MetaMathQA
dataset_config_name=default 


per_device_train_batch_size=4
per_device_eval_batch_size=1
gradient_accumulation_steps=8
gradient_checkpointing=True

max_steps=1000
dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=no
seed=42

log_out=log.out
learning_rates=(5e-5)

for learning_rate in "${learning_rates[@]}"
do
    output_dir=/root/aslora/checkpoints/${model_name_or_path##*/}/codefeedback/${learning_rate}_${optim_notes}
    echo "${output_dir}"
    mkdir -p ${output_dir}

    echo  --model_name_or_path ${model_name_or_path} \
          --output_dir ${output_dir} \
          --dataset_name ${dataset_name} \
          --dataset_config_name ${dataset_config_name} \
          --per_device_train_batch_size ${per_device_train_batch_size} \
          --per_device_eval_batch_size ${per_device_eval_batch_size} \
          --max_steps ${max_steps} \
          --overwrite_output_dir \
          --do_train ${do_train} \
          --do_eval \
          --lr_scheduler_type ${lr_scheduler_type} \
          --block_size ${block_size} \
          --seed ${seed} \
          --fp16 ${fp16} \
          --gradient_checkpointing ${gradient_checkpointing} \
          --local_rank ${local_rank} \
          --dataloader_num_workers ${dataloader_num_workers} \
          --disable_tqdm False \
          --save_strategy ${save_strategy} \
          --evaluation_strategy ${evaluation_strategy} \
          --load_best_model_at_end True \
          --learning_rate ${learning_rate} \
          --optim_notes ${optim_notes} \
          --split_strategy ${split_strategy} \
          --num_rounds ${num_rounds} \
          --num_clients ${num_clients} \
          --sample_clients ${sample_clients} \
          --max_gate_samples ${max_gate_samples} \
          --max_train_samples ${max_train_samples} \
          --gradient_accumulation_steps ${gradient_accumulation_steps} \
          > ${output_dir}/config.txt

    if [ -f ${output_dir}/${log_out} ]; then
        rm -f ${output_dir}/${log_out}
    fi

    python main_lora.py \
        --model_name_or_path ${model_name_or_path} \
        --output_dir ${output_dir} \
        --dataset_name ${dataset_name} \
        --dataset_config_name ${dataset_config_name} \
        --per_device_train_batch_size ${per_device_train_batch_size} \
        --per_device_eval_batch_size ${per_device_eval_batch_size} \
        --max_steps ${max_steps} \
        --overwrite_output_dir \
        --do_train ${do_train} \
        --do_eval \
        --lr_scheduler_type ${lr_scheduler_type} \
        --seed ${seed} \
        --fp16 ${fp16} \
        --gradient_checkpointing ${gradient_checkpointing} \
        --local_rank ${local_rank} \
        --block_size ${block_size} \
        --dataloader_num_workers ${dataloader_num_workers} \
        --disable_tqdm False \
        --save_strategy ${save_strategy} \
        --evaluation_strategy ${evaluation_strategy} \
        --learning_rate ${learning_rate} \
        --optim_notes ${optim_notes} \
        --split_strategy ${split_strategy} \
        --num_rounds ${num_rounds} \
        --num_clients ${num_clients} \
        --sample_clients ${sample_clients} \
        --max_gate_samples ${max_gate_samples} \
        --max_train_samples ${max_train_samples} \
        --gradient_accumulation_steps ${gradient_accumulation_steps}

done
