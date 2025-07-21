# SET visible device
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizers 并行处理
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("=" * 100)

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Optional, Literal
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from prompter import Prompter
from evaluate_grasp import evaluate_model
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_prune', type=int, default=7, help='Number of layers to prune')
    parser.add_argument('--layers_to_remove', type=str, 
                       default="29,30,28,27,24,16,14,25,13,20,21,19,23,17,22,26,15,9,7,12,4,6,1,3,5,11,0,8,2,10,18,31",
                       help='Comma-separated list of layer indices to remove (in priority order)')
    args = parser.parse_args()

    model_name = 'meta-llama/Llama-3.1-8B'
    # 在模型加载时添加量化
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 将字符串参数转换为整数列表
    layers_to_remove_all = [int(x.strip()) for x in args.layers_to_remove.split(',')]
    layers_to_remove = layers_to_remove_all[:args.num_prune]
    device="cuda:0"

    # For simplify, we manually remove the redundant layers found by running run_shortgpt.py
    # remove layers in reverse to avoid indexing errors
    for layer_idx in sorted(layers_to_remove, reverse=True):
        try:
            del model.model.layers[layer_idx]
        except IndexError:
            print(f"layer {layer_idx} does not exist, function may have already been called")
    
    print(layers_to_remove)
    print("=" * 100)


    result = evaluate_model(model, tokenizer, model_name="llama3", tasks="mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq", eval_ppl="wikitext2,ptb", device=device) # boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa
