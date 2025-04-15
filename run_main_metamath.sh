#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

optim_notes=Adamw # this indicates which optim we use
split_strategy=iid
num_rounds=1
num_clients=1
sample_clients=1

max_gate_samples=50
max_train_samples=100000
do_train=True
model_name_or_path="meta-llama/Meta-Llama-3-8B"
# "meta-llama/Llama-3.2-3B"
dataset_name=meta-math/MetaMathQA
dataset_config_name=default
per_device_train_batch_size=2
per_device_eval_batch_size=8
gradient_accumulation_steps=4


# num_train_epochs=1
max_steps=1000
dataloader_num_workers=16
evaluation_strategy=epoch
save_strategy=no
seed=42


log_out=log.out
# learning_rate=3e-6
learning_rates=(1e-4 3e-5 1e-5 3e-6)
# Loop over each learning rate
for learning_rate in "${learning_rates[@]}"
do
      output_dir=/home/yjw5427/aslora_new/checkpoints/${model_name_or_path##*/}/${dataset_name}/${learning_rate}_${optim_notes}


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
            --seed ${seed} \
            --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm False \
            --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
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


      if [ ! -f ${output_dir}/log.out ];then
      echo "The file doesn't exist."
      else
      rm -d ${output_dir}/${log_out}
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
            --seed ${seed} \
            --dataloader_num_workers ${dataloader_num_workers} --disable_tqdm False \
            --save_strategy ${save_strategy} --evaluation_strategy ${evaluation_strategy} \
            --learning_rate ${learning_rate} \
            --optim_notes ${optim_notes} \
            --split_strategy ${split_strategy} \
            --num_rounds ${num_rounds} \
            --num_clients ${num_clients} \
            --sample_clients ${sample_clients} \
            --max_gate_samples ${max_gate_samples} \
            --max_train_samples ${max_train_samples} \
            --gradient_accumulation_steps ${gradient_accumulation_steps} \

done
# cd gpt_moe
# bash run_main_metamath.sh > logm2.txt 

# cp -r /home/yjw5427/Hyper_MoE/gpt-2-moe/transformers/models/bert /home/yjw5427/fedmoe/gpt_moe/transformers/models/bert



# /home/yjw5427/miniconda3/envs/fedmoe/lib/python3.10/site-packages/sentence_transformers/SentenceTransformer.py

# /home/yjw5427/miniconda3/lib/python3.8/site-packages/transformers/models/bert/


# /home/yjw5427/Hyper_MoE/gpt-2-moe/transformers

# mv ./leaf-master /data/yujia/leaf-master

# rm -rf /tmp/tmux-10282/default      very very important

# old transformer 4.31.0

# cd gpt_moe
# bash run_main_moe.sh 

# cp -r /home/yjw5427/Hyper_MoE/gpt-2-moe/transformers/models/bert /home/yjw5427/fedmoe/gpt_moe/transformers/models/bert



# /home/yjw5427/miniconda3/envs/fedmoe/lib/python3.10/site-packages/sentence_transformers/SentenceTransformer.py

# /home/yjw5427/miniconda3/lib/python3.8/site-packages/transformers/models/bert/


# /home/yjw5427/Hyper_MoE/gpt-2-moe/transformers

# mv ./leaf-master /data/yujia/leaf-master

# rm -rf /tmp/tmux-10282/default      very very important

# old transformer 4.31.0

# scp ./Downloads/aasyncFL_nlu.zip yujia_wang@64.181.237.178:~/