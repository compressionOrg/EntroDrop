from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import (
    LlamaModel, LlamaAttention, apply_rotary_pos_emb, repeat_kv, 
    LlamaMLP, LlamaRMSNorm, LlamaConfig, LlamaDecoderLayer
)
import torch.nn as nn
import torch
from typing import Optional, List, Tuple, Dict
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
import math
from tqdm import tqdm
import torch.nn.functional as F
import gc
import numpy as np
from collections import defaultdict
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovedMetrics:
    """改进的层级重要性评估指标类"""
    
    @staticmethod
    def compute_rrc(h_prev: torch.Tensor, h_curr: torch.Tensor, 
                   normalize_by_dim: bool = True) -> float:
        """计算相对残差贡献 (RRC)
        
        Args:
            h_prev: 前一状态的隐藏状态
            h_curr: 当前状态的隐藏状态
            normalize_by_dim: 是否按维度归一化
            
        Returns:
            RRC分数
        """
        # 确保张量在同一设备上并转换为float32以提高精度
        h_prev = h_prev.detach().float()
        h_curr = h_curr.detach().float()
        
        # 计算更新向量
        update = h_curr - h_prev
        
        # 使用Frobenius范数，更稳定
        update_norm = torch.norm(update, p='fro')
        input_norm = torch.norm(h_prev, p='fro')
        
        # 按维度归一化
        if normalize_by_dim:
            dim_factor = math.sqrt(h_prev.numel())
            update_norm = update_norm / dim_factor
            input_norm = input_norm / dim_factor
        
        # 避免除零，使用更大的epsilon
        rrc_score = update_norm / (input_norm + 1e-8)
        
        return rrc_score.item()
    
    @staticmethod
    def compute_cosine_similarity(h_prev: torch.Tensor, h_curr: torch.Tensor) -> float:
        """计算前后状态的余弦相似度"""
        h_prev_flat = h_prev.detach().flatten().float()
        h_curr_flat = h_curr.detach().flatten().float()
        
        cos_sim = F.cosine_similarity(h_prev_flat.unsqueeze(0), 
                                     h_curr_flat.unsqueeze(0))
        return cos_sim.item()
    
    @staticmethod
    def compute_entropy_change(h_prev: torch.Tensor, h_curr: torch.Tensor) -> float:
        """计算隐藏状态的熵变化"""
        def compute_entropy(tensor):
            # 将张量归一化为概率分布
            tensor_flat = tensor.detach().flatten().float()
            # 使用softmax确保为正值
            probs = F.softmax(tensor_flat, dim=0)
            # 计算熵
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            return entropy.item()
        
        entropy_prev = compute_entropy(h_prev)
        entropy_curr = compute_entropy(h_curr)
        
        return abs(entropy_curr - entropy_prev)
    
    @staticmethod
    def compute_gradient_norm(layer: nn.Module) -> float:
        """计算层的梯度范数"""
        total_norm = 0.0
        param_count = 0
        
        for param in layer.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count == 0:
            return 0.0
        
        return math.sqrt(total_norm / param_count)


class LlamaAttentionDropImproved(LlamaAttention):
    """改进的注意力层"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.metrics = ImprovedMetrics()
    
    def calculate_attention(self, query_states, key_states, attention_mask):
        """计算注意力权重"""
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # 使用更稳定的softmax计算
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        
        return attn_weights


class LlamaDecoderLayerDropImproved(LlamaDecoderLayer):
    """改进的解码器层"""
    
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttentionDropImproved(config=config, layer_idx=layer_idx)
        self.layer_idx = layer_idx
        self.metrics = ImprovedMetrics()
        
        # 用于累积统计信息
        self.stats = {
            'rrc_scores': [],
            'cosine_similarities': [],
            'entropy_changes': [],
            'gradient_norms': []
        }
    
    def compute_layer_importance(self, h_prev: torch.Tensor, h_curr: torch.Tensor, 
                               h_attn_out: torch.Tensor) -> Dict[str, float]:
        """计算层的多维重要性指标
        
        Args:
            h_prev: 层输入
            h_curr: 层输出
            h_attn_out: 注意力输出
            
        Returns:
            包含多个重要性指标的字典
        """
        metrics = {}
        
        # 1. RRC分数
        metrics['rrc'] = self.metrics.compute_rrc(h_prev, h_curr, normalize_by_dim=True)
        
        # 2. 余弦相似度 (1 - similarity 表示变化程度)
        cos_sim = self.metrics.compute_cosine_similarity(h_prev, h_curr)
        metrics['cosine_change'] = 1.0 - cos_sim
        
        # 3. 熵变化
        metrics['entropy_change'] = self.metrics.compute_entropy_change(h_prev, h_curr)
        
        # 4. 注意力贡献
        if h_attn_out is not None:
            metrics['attn_rrc'] = self.metrics.compute_rrc(h_prev, h_attn_out, normalize_by_dim=True)
        else:
            metrics['attn_rrc'] = 0.0
        
        # 5. 梯度范数 (如果在训练模式)
        if self.training:
            metrics['gradient_norm'] = self.metrics.compute_gradient_norm(self)
        else:
            metrics['gradient_norm'] = 0.0
        
        return metrics
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        drop_attentions = None,
        drop_layer = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings = None,
        **kwargs,
    ):
        """改进的前向传播"""
        if drop_layer:
            # 返回空的指标
            empty_metrics = {
                'rrc': 0.0,
                'cosine_change': 0.0,
                'entropy_change': 0.0,
                'attn_rrc': 0.0,
                'gradient_norm': 0.0
            }
            return (hidden_states,), empty_metrics
        
        orig_input = hidden_states.clone()  # 使用clone避免引用问题
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        h_attn_out = None
        if not drop_attentions:
            # Self Attention
            hidden_states, self_attn_weights, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            h_attn_out = hidden_states.clone()
            hidden_states = residual + hidden_states
        else:
            hidden_states = residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # 计算多维重要性指标
        importance_metrics = self.compute_layer_importance(orig_input, hidden_states, h_attn_out)
        
        # 累积统计信息
        for key, value in importance_metrics.items():
            if key in self.stats:
                self.stats[key].append(value)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs, importance_metrics


class LayerImportanceRanker:
    """层重要性排序器"""
    
    def __init__(self, config):
        self.config = config
        self.num_layers = config.num_hidden_layers
        
        # 权重配置 - 可以根据实验调整
        self.metric_weights = {
            'rrc': 0.4,
            'cosine_change': 0.2,
            'entropy_change': 0.2,
            'attn_rrc': 0.1,
            'gradient_norm': 0.1
        }
    
    def compute_composite_score(self, metrics_list: List[Dict[str, float]]) -> List[float]:
        """计算复合重要性分数
        
        Args:
            metrics_list: 每层的指标字典列表
            
        Returns:
            每层的复合分数列表
        """
        composite_scores = []
        
        # 首先归一化每个指标
        normalized_metrics = self._normalize_metrics(metrics_list)
        
        for layer_metrics in normalized_metrics:
            score = 0.0
            for metric_name, weight in self.metric_weights.items():
                if metric_name in layer_metrics:
                    score += weight * layer_metrics[metric_name]
            composite_scores.append(score)
        
        return composite_scores
    
    def _normalize_metrics(self, metrics_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """归一化指标到[0,1]范围"""
        if not metrics_list:
            return []
        
        # 收集所有指标的最大最小值
        metric_ranges = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
        
        for layer_metrics in metrics_list:
            for metric_name, value in layer_metrics.items():
                if not math.isnan(value) and not math.isinf(value):
                    metric_ranges[metric_name]['min'] = min(metric_ranges[metric_name]['min'], value)
                    metric_ranges[metric_name]['max'] = max(metric_ranges[metric_name]['max'], value)
        
        # 归一化
        normalized_list = []
        for layer_metrics in metrics_list:
            normalized = {}
            for metric_name, value in layer_metrics.items():
                min_val = metric_ranges[metric_name]['min']
                max_val = metric_ranges[metric_name]['max']
                
                if max_val - min_val > 1e-10:  # 避免除零
                    normalized[metric_name] = (value - min_val) / (max_val - min_val)
                else:
                    normalized[metric_name] = 0.0
            
            normalized_list.append(normalized)
        
        return normalized_list
    
    def rank_layers(self, metrics_list: List[Dict[str, float]], 
                   protect_layers: Optional[List[int]] = None) -> List[int]:
        """对层进行重要性排序
        
        Args:
            metrics_list: 每层的指标字典列表
            protect_layers: 需要保护的层索引列表
            
        Returns:
            层的重要性排序（从最不重要到最重要）
        """
        if protect_layers is None:
            # 默认保护前两层和最后一层
            protect_layers = [0, 1, self.num_layers - 1]
        
        # 计算复合分数
        composite_scores = self.compute_composite_score(metrics_list)
        
        # 为保护的层设置极高分数
        for layer_idx in protect_layers:
            if 0 <= layer_idx < len(composite_scores):
                composite_scores[layer_idx] = float('inf')
        
        # 排序：分数低的层排在前面（更容易被剪枝）
        sorted_indices = sorted(range(len(composite_scores)), 
                              key=lambda i: (composite_scores[i], i))
        
        # 转换为排名
        ranks = [0] * len(composite_scores)
        for rank, idx in enumerate(sorted_indices):
            ranks[idx] = rank
        
        return ranks


class LlamaModelDropImproved(LlamaModel):
    """改进的Llama模型"""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList([
            LlamaDecoderLayerDropImproved(config, i) 
            for i in range(config.num_hidden_layers)
        ])
        self.layer_ranker = LayerImportanceRanker(config)
        
        # 统计信息
        self.layer_stats = defaultdict(list)
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        """改进的前向传播"""
        # 标准的模型初始化
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask=attention_mask, 
            input_tensor=inputs_embeds, 
            cache_position=cache_position, 
            past_key_values=past_key_values, 
            output_attentions=output_attentions
        )
        hidden_states = inputs_embeds

        # 创建位置嵌入
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 解码器层
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        
        layer_importance_metrics = []

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            layer_outputs, importance_metrics = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                drop_attentions=False,
                drop_layer=False,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            layer_importance_metrics.append(importance_metrics)
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        return output, layer_importance_metrics


class LlamaForCausalLMDropImproved(LlamaForCausalLM):
    """改进的因果语言模型"""
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = LlamaModelDropImproved(config)
        
        # 累积的重要性指标
        self.accumulated_metrics = []
        
        logger.info(f"Initialized improved model with dtype: {self.model.dtype}")
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ):
        """改进的前向传播"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = model_outputs[0].last_hidden_state
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output_tuple = (logits,) + model_outputs[1:]
            return (loss,) + output_tuple if loss is not None else output_tuple

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=model_outputs[0].past_key_values,
            hidden_states=model_outputs[0].hidden_states,
            attentions=model_outputs[0].attentions,
        ), model_outputs[1]  # 返回重要性指标
    
    def process_layers_improved(self, dataloader, num_samples: int = None) -> Tuple[List[int], Dict]:
        """改进的层处理方法
        
        Args:
            dataloader: 数据加载器
            num_samples: 限制处理的样本数量
            
        Returns:
            层排序和详细统计信息
        """
        logger.info("开始处理层重要性评估...")
        
        # 累积所有批次的指标
        all_batch_metrics = []
        
        # 设置为评估模式
        self.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                if num_samples and batch_idx >= num_samples:
                    break
                
                try:
                    inputs = {"input_ids": batch[0].cuda()}
                    
                    # 前向传播
                    outputs, layer_metrics = self.forward(**inputs, output_attentions=False)
                    
                    all_batch_metrics.append(layer_metrics)
                    
                    # 清理GPU内存
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.warning(f"Batch {batch_idx} failed: {e}")
                    continue
        
        # 平均所有批次的指标
        averaged_metrics = self._average_metrics(all_batch_metrics)
        
        # 使用改进的排序器
        layer_ranks = self.model.layer_ranker.rank_layers(averaged_metrics)
        
        # 生成详细统计
        stats = self._generate_detailed_stats(averaged_metrics, layer_ranks)
        
        # 保存到配置
        self.config.drop_layers_order = layer_ranks
        
        logger.info(f"层重要性排序完成: {layer_ranks}")
        
        return layer_ranks, stats
    
    def _average_metrics(self, all_batch_metrics: List[List[Dict]]) -> List[Dict[str, float]]:
        """平均所有批次的指标"""
        if not all_batch_metrics:
            return []
        
        num_layers = len(all_batch_metrics[0])
        averaged = []
        
        for layer_idx in range(num_layers):
            layer_metrics = defaultdict(list)
            
            # 收集该层在所有批次中的指标
            for batch_metrics in all_batch_metrics:
                if layer_idx < len(batch_metrics):
                    for metric_name, value in batch_metrics[layer_idx].items():
                        if not math.isnan(value) and not math.isinf(value):
                            layer_metrics[metric_name].append(value)
            
            # 计算平均值
            averaged_layer = {}
            for metric_name, values in layer_metrics.items():
                if values:
                    averaged_layer[metric_name] = sum(values) / len(values)
                else:
                    averaged_layer[metric_name] = 0.0
            
            averaged.append(averaged_layer)
        
        return averaged
    
    def _generate_detailed_stats(self, metrics: List[Dict], ranks: List[int]) -> Dict:
        """生成详细的统计信息"""
        stats = {
            'layer_metrics': metrics,
            'layer_ranks': ranks,
            'summary': {
                'most_important_layers': [],
                'least_important_layers': [],
                'metric_correlations': {}
            }
        }
        
        # 找出最重要和最不重要的层
        rank_layer_pairs = [(rank, idx) for idx, rank in enumerate(ranks)]
        rank_layer_pairs.sort()
        
        stats['summary']['least_important_layers'] = [idx for _, idx in rank_layer_pairs[:5]]
        stats['summary']['most_important_layers'] = [idx for _, idx in rank_layer_pairs[-5:]]
        
        # 计算指标相关性
        if metrics:
            metric_names = list(metrics[0].keys())
            for i, metric1 in enumerate(metric_names):
                for j, metric2 in enumerate(metric_names[i+1:], i+1):
                    values1 = [m[metric1] for m in metrics]
                    values2 = [m[metric2] for m in metrics]
                    
                    # 简单的皮尔逊相关系数
                    if len(values1) > 1:
                        corr = np.corrcoef(values1, values2)[0, 1]
                        if not np.isnan(corr):
                            stats['summary']['metric_correlations'][f'{metric1}_vs_{metric2}'] = corr
        
        return stats
    
    def save_analysis_results(self, stats: Dict, filepath: str):
        """保存分析结果"""
        import json
        
        # 转换numpy类型为Python原生类型
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        converted_stats = convert_types(stats)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"分析结果已保存到: {filepath}")