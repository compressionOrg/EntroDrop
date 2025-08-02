# SET visible device
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用 tokenizers 并行处理

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
from dataset.loader import get_test_data
import argparse
import logging
import sys
import time
import itertools
import copy

def count_parameters(model):
    """计算模型的参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        int: 模型的总参数量
    """
    return sum(p.numel() for p in model.parameters())

def format_parameters(param_count):
    """将参数量格式化为以B为单位的字符串
    
    Args:
        param_count (int): 参数数量
        
    Returns:
        str: 格式化后的参数量字符串
    """
    return f"{param_count / 1e9:.2f}B"

def count_compression_ratio(original_model, compressed_model):
    """计算压缩比例
    
    Args:
        original_model: 原始模型
        compressed_model: 压缩后的模型
    """
    compressed_total_params = sum(p.numel() for p in compressed_model.parameters())
    original_total_params = sum(p.numel() for p in original_model.parameters())

    compression_ratio = 1 - compressed_total_params / original_total_params
    print(f'Total Params: {original_total_params} || Compressed Params: {compressed_total_params} || Compression Ratio: {compression_ratio:.3f}')

@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=2048, batch_size=16, device="cuda"):
    """评估模型推理效率
    
    Args:
        model: 要评估的模型
        tokenizer: 分词器
        dataset: 数据集名称
        original_len: 原始序列长度
        generated_len: 生成序列长度
        batch_size: 批次大小
        device: 设备
    """
    model.to(device)
    model.eval()
    throughput = 0
    token_num = 0
    end_memory = 0
    num_batches_to_fetch = 10
    test_loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size = batch_size)
    weight_memory = torch.cuda.memory_allocated()
    for batch_idx, batch_data in enumerate(itertools.islice(test_loader, num_batches_to_fetch)):
        batch = batch_data.to(device)
        token_num += batch.shape[0] * generated_len
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        start_time = time.time()
        generation_output = model.generate(
                input_ids=batch,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                use_cache=True,
                top_k=50,
                max_length=original_len+generated_len,
                top_p=0.95,
                temperature=1,
        )
        torch.cuda.synchronize()
        end_time = time.time()
        end_memory = max(torch.cuda.max_memory_allocated(0), end_memory)
        if torch.isfinite(generation_output[0]).all():  # check if the generation is successful since fp16 may cause nan
            throughput += end_time - start_time
            print("time: {}".format(end_time - start_time))
    logger.info("Total Memory: {} GB".format(end_memory/(1024 ** 3)))
    logger.info("Weight Memory: {} GB".format(weight_memory/(1024 ** 3)))
    logger.info("Activation Memory: {} GB".format((end_memory - start_memory)/(1024 ** 3)))
    logger.info("Throughput: {} tokens/sec".format(token_num / throughput))
    model.to('cpu')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B',
                    choices=['meta-llama/Llama-3.1-8B', 'meta-llama/Llama-2-7b-hf', 
                            'mistralai/Mistral-7B-v0.3', 'baichuan-inc/Baichuan2-7B-Base'],
                    help='Model name to use for layer removal (default: meta-llama/Llama-3.1-8B)')
    parser.add_argument('--num_prune', type=int, default=7, help='Number of layers to prune')
    parser.add_argument('--layers_order', type=str, 
                       default="29,30,28,27,24,16,14,25,13,20,21,19,23,17,22,26,15,9,7,12,4,6,1,3,5,11,0,8,2,10,18,31",
                       help='Comma-separated list of layer indices to remove (in priority order)')
    parser.add_argument('--tasks', type=str, 
                       default="mathqa,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa,boolq",
                       help='Comma-separated list of evaluation tasks')
    parser.add_argument('--eval_ppl', type=str, 
                       default="wikitext2,ptb",
                       help='Comma-separated list of perplexity evaluation datasets')
    parser.add_argument('--log_file', type=str, default=None, help='Path to log file for saving program output')
    parser.add_argument('--run_inference_eval', action='store_true', help='Run inference efficiency evaluation')
    parser.add_argument('--eval_original_model', action='store_true', help='Evaluate original model inference efficiency (optional)')
    parser.add_argument('--inference_dataset', type=str, default='wikitext2', help='Dataset for inference evaluation')
    parser.add_argument('--original_len', type=int, default=4, help='Original sequence length for inference')
    parser.add_argument('--generated_len', type=int, default=2048, help='Generated sequence length for inference')
    parser.add_argument('--inference_batch_size', type=int, default=16, help='Batch size for inference evaluation')

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


    model_name = args.model_name
    logger.info(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 保存原始模型的副本用于比较
    original_model = copy.deepcopy(model)
    
    # 计算原始模型参数量
    original_param_count = count_parameters(model)
    logger.info(f"Original model parameters: {format_parameters(original_param_count)} ({original_param_count:,})")
    
    logger.info(f"Layers order: {args.layers_order}")
    logger.info(f"Num prune: {args.num_prune}")
    
    # 将字符串参数转换为整数列表
    layers_to_remove_all = [int(x.strip()) for x in args.layers_order.split(',')]
    layers_to_remove = layers_to_remove_all[:args.num_prune]
    device="cuda:0"

    # For simplify, we manually remove the redundant layers found by running run_shortgpt.py
    # remove layers in reverse to avoid indexing errors
    for layer_idx in sorted(layers_to_remove, reverse=True):
        try:
            del model.model.layers[layer_idx]
        except IndexError:
            logger.warning(f"layer {layer_idx} does not exist, function may have already been called")
    
    logger.info(f"Layers to remove: {layers_to_remove}")
    
    # 计算剪枝后模型参数量
    pruned_param_count = count_parameters(model)
    pruning_ratio = (original_param_count - pruned_param_count) / original_param_count * 100
    
    logger.info(f"Pruned model parameters: {format_parameters(pruned_param_count)} ({pruned_param_count:,})")
    logger.info(f"Parameters reduced: {format_parameters(original_param_count - pruned_param_count)} ({original_param_count - pruned_param_count:,})")
    logger.info(f"Pruning ratio: {pruning_ratio:.2f}%")
    logger.info("=" * 100)


    # result = evaluate_model(model, tokenizer, model_name="llama3", tasks=args.tasks, eval_ppl=args.eval_ppl, device=device, log_file=args.log_file)
    
    # 运行推理效率评估（如果启用）
    if args.run_inference_eval:
        logger.info("Starting inference efficiency evaluation...")
        logger.info("=" * 50)
        
        # 计算压缩比例
        logger.info("Compression ratio analysis:")
        count_compression_ratio(original_model, model)
        
        # 评估原始模型的推理效率（可选）
        if args.eval_original_model:
            logger.info("\nRunning inference efficiency evaluation on original model...")
            eff_eval(original_model, tokenizer, 
                    dataset=args.inference_dataset,
                    original_len=args.original_len, 
                    generated_len=args.generated_len,
                    batch_size=args.inference_batch_size, 
                    device=device)
        
        logger.info("\nRunning inference efficiency evaluation on pruned model...")
        eff_eval(model, tokenizer, 
                dataset=args.inference_dataset,
                original_len=args.original_len, 
                generated_len=args.generated_len,
                batch_size=args.inference_batch_size, 
                device=device)
        
        logger.info("Inference efficiency evaluation completed.")
        logger.info("=" * 50)
