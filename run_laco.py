# from https://github.com/yangyifei729/LaCo/blob/main/laco_llama-13b.ipynb
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizers 并行处理

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_grasp import evaluate_model
import argparse
import logging
import sys


def merge_layers_return_model(model, merge_base_lay, merge_layer_num):
   
    merge_layer_num = min(merge_layer_num, len(model.model.layers) - merge_base_lay - 1)
    model_copy = deepcopy(model)
    
    for diff_lay in range(merge_base_lay+1, merge_base_lay+1+merge_layer_num):      
        # gate_proj
        model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.gate_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.gate_proj.weight.data
        )
        # down_proj
        model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.down_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.down_proj.weight.data
        )
        # up_proj
        model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data.add_(
            model.model.layers[diff_lay].mlp.up_proj.weight.data - model_copy.model.layers[merge_base_lay].mlp.up_proj.weight.data
        ) 

        # q_proj
        model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.q_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.q_proj.weight.data
        )

        # k_proj
        model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.k_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.k_proj.weight.data
        ) 
    
        # v_proj
        model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.v_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.v_proj.weight.data
        )
    
        # o_proj
        model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data.add_(
            model.model.layers[diff_lay].self_attn.o_proj.weight.data - model_copy.model.layers[merge_base_lay].self_attn.o_proj.weight.data
        )        
                       
    for diff_lay in range(merge_base_lay+merge_layer_num, merge_base_lay, -1):
        del(model_copy.model.layers[diff_lay])

    return model_copy


def cal_last_hidden_sim(model1, model2, tokenizer, sents, device1, device2, logger=None):
    model1.to(device1)
    model2.to(device2)
    sim_ls = []
    for s in sents:
        encoded_inputs = tokenizer(s, return_tensors='pt')
        with torch.no_grad():
            encoded_inputs = encoded_inputs.to(model1.device)
            outputs1 = model1(**encoded_inputs, output_hidden_states=True)
        hidden_states1: torch.Tensor = outputs1.hidden_states[-1] # (1, seq_len, hidden)
        with torch.no_grad():
            encoded_inputs = encoded_inputs.to(model2.device)
            outputs2 = model2(**encoded_inputs, output_hidden_states=True)
        hidden_states2: torch.Tensor = outputs2.hidden_states[-1] # (1, seq_len, hidden)
        sim_ls.append(torch.cosine_similarity(hidden_states1.cpu().squeeze(0).flatten().unsqueeze(0), hidden_states2.cpu().squeeze(0).flatten().unsqueeze(0)))
    sim_ls = [i.item() for i in sim_ls]
    if logger:
        logger.debug(f"Similarity scores: {sim_ls}, Mean: {np.mean(sim_ls):.4f}")
    return np.mean(sim_ls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merge_layers', type=int, default=4, help='Number of layers to merge')
    parser.add_argument('--log_file', type=str, default=None, help='Path to log file for saving program output')
    args = parser.parse_args()

    # Set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent log propagation to the root logger
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Log to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Log to file
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    model_name = 'meta-llama/Llama-3.1-8B'
    logger.info(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model_copy_to_compress = deepcopy(model)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    device1 = "cuda:0"
    device2 = "cuda:2"
    
    logger.info(f"Merge layers: {args.merge_layers}")
    logger.info(f"Using devices: {device1}, {device2}")

    INTERVAL = 1
    MERGE_LAYERS = args.merge_layers
    HIGHEST_LAY = 30
    LOWEST_LAY = 2
    THRESHOLD = 0.45

    lay = HIGHEST_LAY - MERGE_LAYERS
    sents = []
    en_wiki_selected = [
        'Mouron () is a commune in the Arde',
        'The 81st Mechanised Brigade () is a mechanised brigade of the Romanian Land Force',
        'There are 18 National Natural Landmarks in the U.S. state of Washington, out of nearly',
        'Torreorgaz is a municipality in the',
        'Copa Libertadores 1973 was won by defending champions Independiente of A'
    ]
    sents.extend(en_wiki_selected)

    while lay >= LOWEST_LAY:
        logger.info(f"Processing layer: {lay}")
        logger.info(f"Current model layers: {len(model_copy_to_compress.model.layers)}")
        model_copy_to_compress.cpu() # move to cpu first
        tmp_merged_model = merge_layers_return_model(model_copy_to_compress, lay, MERGE_LAYERS-1)
        sim_value = cal_last_hidden_sim(model, tmp_merged_model, tokenizer, sents, device1=device1, device2=device2, logger=logger)
        logger.info(f"Similarity value: {sim_value:.4f}")
        if sim_value > THRESHOLD:
            logger.info(f"Similarity {sim_value:.4f} > threshold {THRESHOLD}, merging layers")
            model_copy_to_compress = tmp_merged_model
            lay -= INTERVAL
            if lay >= len(model_copy_to_compress.model.layers):
                lay = len(model_copy_to_compress.model.layers) - 1 - MERGE_LAYERS
        else:
            logger.info(f"Similarity {sim_value:.4f} <= threshold {THRESHOLD}, moving to next layer")
            lay -= 1
    
    # empty cache
    model.to(device="cpu")
    model_copy_to_compress.to(device="cpu")
    if "cuda" in device1 or "cuda" in device2:
        torch.cuda.empty_cache()

    logger.info(f"Final compressed model: {len(model_copy_to_compress.model.layers)} layers")
    logger.info("=" * 100)
    
    result = evaluate_model(model_copy_to_compress, tokenizer, model_name="llama", tasks="mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq", eval_ppl="wikitext2", device=device1, log_file=args.log_file)
