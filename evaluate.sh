#!/bin/bash

set -e
model_path=/home/zhangyingying/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/model
order="29_30_28_26_18_16_17_24_6_20_22_4_19_15_25_27_23_7_8_0_3_13_10_5_14_21_1_12_2_9_11_31"

for drop_num in 8 12; do
    echo "Drop layer: ${drop_num}"
    CUDA_VISIBLE_DEVICES=0 lm_eval --model hf --model_args pretrained=${model_path},device_map=auto,dtype=float16,drop_layers_order=${order},drop_layers=${drop_num} --tasks piqa,hellaswag,winogrande,wsc273,commonsense_qa,mmlu,cmmlu,arc_easy,arc_challenge,openbookqa,race --batch_size auto --trust_remote_code
done

