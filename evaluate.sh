#!/bin/bash

set -e
model_path=YourModelPathHere
orders=('"29_30_28_27_26_23_25_24_21_22_18_17_19_20_16_15_14_13_10_2_3_12_7_5_8_0_6_9_4_1_11_31"')

for drop_num in 8 12; do
    echo "Drop layer: ${drop_num}"
    CUDA_VISIBLE_DEVICES=0 lm_eval --model hf --model_args  pretrained=${model_path},parallelize=True,dtype=float16,drop_layers_order=${order},drop_layers=${drop_num} --tasks piqa,hellaswag,winogrande,wsc273,commonsense_qa,mmlu,cmmlu,arc_easy,arc_challenge,openbookqa,race --batch_size auto --trust_remote_code
done

