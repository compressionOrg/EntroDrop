# from https://github.com/sramshetty/ShortGPT

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset.loader import get_calibration_dataloader
from evaluate_grasp import evaluate_model
from typing import Optional, List, Literal
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import argparse


def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular: bool = False,
    metric: str = "normalized_combo",
    alpha: float = 0.5,
):
    """
    input_hidden_state: B, S, D
    output_hidden_state: B, S, D
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state_flat = input_hidden_state.reshape(-1, d)
    output_hidden_state_flat = output_hidden_state.reshape(-1, d)

    if metric == "normalized_combo":
        # MSE part
        mse = torch.mean((input_hidden_state_flat - output_hidden_state_flat) ** 2, dim=-1)
        mse_norm = torch.sigmoid(mse)

        # Cosine similarity part
        sim = F.cosine_similarity(input_hidden_state_flat, output_hidden_state_flat, dim=-1).nan_to_num(nan=0.5)
        cos_sim_term = 1 - sim
        cos_sim_term_norm = cos_sim_term / 2.0  # Scale from [0, 2] to [0, 1]

        return alpha * mse_norm + (1 - alpha) * cos_sim_term_norm

    # Original logic for cosine and angular, refactored for efficiency
    sim = F.cosine_similarity(input_hidden_state_flat, output_hidden_state_flat, dim=-1).nan_to_num(nan=0.5)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    return 1 - sim

@torch.inference_mode()
def compute_bi(
        model,
        num_prune_layers: Optional[int] = 1,
        calibration_dataloader: Optional[DataLoader] = None,
        hiddens: Optional[List[torch.Tensor]] = None,
        angular: bool = False,
        metric: str = "cosine",
        device: Literal["cpu", "cuda"] = "cuda",
        alpha: float = 0.5,
        *args, **kwargs
    ):
    layer_importances = [0 for _ in model.model.layers]
    """
    Computes layer-wise importances over input tokens.
    """
    def compute_bi_hiddens(hiddens: Optional[List[torch.Tensor]] = None):
        if not angular:
            num_prune_layers = 1

        for i in range(len(hiddens) - num_prune_layers):
            in_hidden = hiddens[i]
            out_hidden = hiddens[i+num_prune_layers]
            if angular:
                # use only last token for angular distance as described in section 3.2
                # https://arxiv.org/pdf/2403.17887.pdf
                in_hidden = in_hidden[:,-1:]
                out_hidden = out_hidden[:,-1:]
            
            layer_importances[i] += block_influence(
                in_hidden,
                out_hidden,
                angular=angular,
                metric=metric,
                alpha=alpha
            ).mean().cpu().item()

    print(f"\n=======>Compute Block Influence")
    assert hiddens is not None or calibration_dataloader is not None, "please provide hidden_states or calibration dataloader to compute block influence"
    if hiddens is not None:
        compute_bi_hiddens(hiddens=hiddens)
    else:
        # 获取模型的第一个参数所在设备，用于多GPU环境下的设备检测
        model_device = next(model.parameters()).device
        
        for batch in tqdm(calibration_dataloader, desc="Compute BI", total=len(calibration_dataloader), leave=True):
            if len(batch) == 2:
                attention_mask = None
            else:
                # 将数据移动到模型所在的设备（对于多GPU模型，通常是第一个GPU）
                attention_mask = batch["attention_mask"].to(device=model_device)
            input_ids = batch["input_ids"].to(device=model_device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, output_hidden_states=True, return_dict=True)
            hiddens = outputs.hidden_states

            compute_bi_hiddens(hiddens=hiddens)
    
    if angular:
        start_layer = np.argsort(np.array(layer_importances[:-num_prune_layers+1]))[0]
        layers_to_remove = list(range(start_layer, start_layer + num_prune_layers))
    else:
        layers_to_remove = np.argsort(np.array(layer_importances))[:num_prune_layers].tolist()
    
    return layer_importances, layers_to_remove

def remove_layers(model, layers_to_remove: Optional[List[int]] = [], layer_importances: Optional[List[float]] = [], angular: Optional[bool] = False, num_prune_layers: Optional[int] = None):
    if not layers_to_remove:
        if angular:
            assert layer_importances, "Need to compute importances with compute_bi(model)"
            assert num_prune_layers, "Need number of layers to prune"
            start_layer = np.argsort(np.array(layer_importances[:-num_prune_layers+1]))[0]
            layers_to_remove = list(range(start_layer, start_layer + num_prune_layers))
        else:
            layers_to_remove = np.argsort(np.array(layer_importances))[:num_prune_layers].tolist()

    if layers_to_remove is not None:
        # remove layers in reverse to avoid indexing errors
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                del model.model.layers[layer_idx]
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
                return []
        
        return layers_to_remove
    else:
        raise NotImplementedError("lack layers_to_remove")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ShortGPT with MSE and alpha parameter')
    parser.add_argument('--alpha', type=float, default=0.8, help='Weight factor for MSE term (default: 0.5)')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.1-8B', 
                        help='Model name to use for pruning (default: baichuan-inc/Baichuan2-7B-Base)') # meta-llama/Llama-3.1-8B', 'mistralai/Mistral-7B-v0.3',  'meta-llama/Llama-2-7b-hf', 'baichuan-inc/Baichuan2-7B-Base'
    parser.add_argument('--save_model', action='store_true', help='Whether to save the pruned model (default: False)')
    parser.add_argument('--num_prune_layers', type=int, default=9, help='Number of layers to prune (default: 9)')
    args = parser.parse_args()

    # 检测可用的GPU数量
    num_gpus = torch.cuda.device_count()
    print(f"检测到 {num_gpus} 个GPU")
    
    # 配置多GPU设备映射
    if num_gpus > 1:
        # 为多GPU环境优化device_map
        device_map = "auto"  # 让transformers自动分配层到不同GPU
        print(f"使用多GPU模式，自动分配模型层到 {num_gpus} 个GPU")
    elif num_gpus == 1:
        device_map = "cuda:0"
        print("使用单GPU模式")
    else:
        device_map = "cpu"
        print("未检测到GPU，使用CPU模式")
    
    model_name = args.model_name
    # 加载模型时使用优化的device_map配置
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        device_map=device_map,
        torch_dtype=torch.float16 if num_gpus > 0 else torch.float32,  # 使用半精度以节省显存
        low_cpu_mem_usage=True  # 减少CPU内存使用
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if num_gpus > 0 else "cpu"
    num_prune_layers = args.num_prune_layers
    calibration_dataloader = get_calibration_dataloader(dataset_name="wikitext2", tokenizer=tokenizer, num_samples=512, batch_size=1, seq_len=2048, padding="max_length")
    # 注意：当使用device_map时，不需要手动调用model.to(device)，因为模型已经分布在指定设备上
    model.eval()

    layer_importances, layers_to_remove = compute_bi(model=model, num_prune_layers=num_prune_layers, angular=False, metric="normalized_combo", calibration_dataloader=calibration_dataloader, device=device, alpha=args.alpha)

    all_layers_removal_order = np.argsort(np.array(layer_importances)).tolist()
    print(f"All layers removal order: {','.join(map(str, all_layers_removal_order))}")

    remove_layers(model=model, layers_to_remove=layers_to_remove, layer_importances=layer_importances, angular=False)

    print(f"remove layers: {layers_to_remove}")
    # print(model)
    
    # Update model config to reflect the actual number of layers after pruning
    if args.save_model:
        original_num_layers = model.config.num_hidden_layers
        new_num_layers = original_num_layers - num_prune_layers
        model.config.num_hidden_layers = new_num_layers
        print(f"Updated num_hidden_layers from {original_num_layers} to {new_num_layers}")
        
        model_name = model_name.replace('/', '-')
        model.save_pretrained(f'{model_name}_shortgpt_l1_layers{num_prune_layers}_alpha{args.alpha}')
        tokenizer.save_pretrained(f'{model_name}_shortgpt_l1_layers{num_prune_layers}_alpha{args.alpha}')
        print(f"Model saved to {model_name}_shortgpt_l1_layers{num_prune_layers}_alpha{args.alpha}")
    else:
        print("Model saving skipped (use --save_model to enable).")

    # result = evaluate_model(model, tokenizer, model_name="llama", tasks="coqa", eval_ppl="", device=device) # boolq,piqa,hellaswag,winogrande,arc_easy,arc_challenge,openbookqa
