from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers.models.llama.modeling_llama import LlamaModel, LlamaAttention, apply_rotary_pos_emb, repeat_kv, LlamaMLP, LlamaRMSNorm, LlamaConfig, LlamaMLP,LlamaDecoderLayer
import torch.nn as nn
import torch
from typing import Optional
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
import math
from tqdm import tqdm
import torch.nn.functional as F
import gc
from scipy.special import gammaln
from sklearn.neighbors import NearestNeighbors
import numpy as np
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import gaussian_kde


class LlamaAttentionDrop(LlamaAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx

    def calculate_attention(self, query_states, key_states, attention_mask):
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float16).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)  
        return attn_weights

    
    def forwardDrop(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        drop_attentions = None,
        last_key_states = None,
        last_query_states = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings = None,  # will become mandatory in v4.46
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()


        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:

            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        device = query_states.device
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos.to(device), sin.to(device))
  

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = self.calculate_attention(query_states, key_states, attention_mask)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, query_states



class LlamaDecoderLayerDrop(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttentionDrop(config=config, layer_idx=layer_idx)
        self.layer_idx = layer_idx


    def estimate_entropy_knn_pytorch(self, data, k=5):
        """
        使用 PyTorch GPU 并行化 kNN 熵估计
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size, sequence_length, hidden_dim = data.shape
        flattened_data = data.reshape(-1, hidden_dim).float().to(device)

        # 🔧 计算距离矩阵（L2 距离）
        dist_matrix = torch.cdist(flattened_data, flattened_data, p=2)

        # 🔧 取 k+1 最小距离（去掉自己）
        knn_distances, _ = torch.topk(dist_matrix, k + 1, largest=False)
        radii = knn_distances[:, 1:]  # 去掉自身距离

        avg_log_radius = torch.mean(torch.log(radii[:, -1] + 1e-10))
        n = flattened_data.shape[0]
        d = hidden_dim

        # 🔧 熵计算（数值稳定）
        entropy = (d * avg_log_radius) + (d / 2) * torch.log(torch.tensor(np.pi)) - torch.lgamma(torch.tensor(d / 2 + 1)) + torch.log(torch.tensor(n)) - torch.log(torch.tensor(k))
        return entropy.item()


    def compute_information_gain_fixed(self, X, Y, k=25):
        """
        修正后的信息增益计算 IG = H(X + f(X)) - H(X)
        """
        entropy_X = self.estimate_entropy_knn_pytorch(X, k)
        entropy_X_fX = self.estimate_entropy_knn_pytorch(Y, k)
        
        information_gain = entropy_X_fX - entropy_X
        return information_gain, entropy_X_fX


    def compute_entropy_marginal_binning_gpu_parallel(self, tensor, num_bins=10):
        """
        使用 PyTorch 在 GPU 上进行多维并行分桶计算熵
        :param tensor: 输入张量 (batch_size, seq_length, embed_dim)
        :param num_bins: 每个维度的分桶数
        :return: 总熵
        """
        device = torch.device("cuda:0")
        tensor = tensor.to(device)

        batch_size, seq_length, embed_dim = tensor.shape

        # 展平数据 (batch_size * seq_length, embed_dim)
        flattened_data = tensor.view(-1, embed_dim)  # (N, D)

        # 并行计算每个维度的最小值和最大值
        min_vals, _ = flattened_data.min(dim=0, keepdim=True)  # (1, D)
        max_vals, _ = flattened_data.max(dim=0, keepdim=True)  # (1, D)

        # 创建每个维度的分桶边界（批量生成）
        bin_edges = torch.linspace(0, 1, num_bins + 1, device=device).unsqueeze(1)  # (B+1, 1)
        bin_edges = min_vals + bin_edges * (max_vals - min_vals)  # (B+1, D)

        # 使用广播机制将数据与分桶边界比较（并行化分桶）
        expanded_data = flattened_data.unsqueeze(0)  # (1, N, D)
        expanded_edges = bin_edges.unsqueeze(1)      # (B+1, 1, D)

        # 数据点分桶（digitized）
        digitized = torch.sum(expanded_data >= expanded_edges[:-1], dim=0) - 1  # (N, D)

        # 计算每个维度的分桶频数（批量计算）
        one_hot_bins = torch.nn.functional.one_hot(digitized, num_bins).float()  # (N, D, B)
        bin_counts = one_hot_bins.sum(dim=0)  # (D, B)

        # 计算概率密度
        probs = bin_counts / bin_counts.sum(dim=1, keepdim=True)  # (D, B)
        probs = torch.clamp(probs, min=1e-10)  # 避免 log(0)

        # 并行计算每个维度的熵
        entropy_per_dim = -torch.sum(probs * torch.log(probs), dim=1)  # (D,)

        # 累加所有维度的熵
        total_entropy = entropy_per_dim.sum()

        return total_entropy.item(), entropy_per_dim.cpu().numpy()


    def compute_information_gain_marginal_binning(self, X, Y, num_bins=20):
        """
        计算信息增益：IG = H(X + f(X)) - H(X)
        :param X: 原始张量 (batch_size, seq_length, embed_dim)
        :param f_X: 变换后的张量 (batch_size, seq_length, embed_dim)
        :param num_bins: 每个维度的分桶数
        :return: 信息增益
        """
        entropy_X, entropy_X_d = self.compute_entropy_marginal_binning_gpu_parallel(X, num_bins)
        entropy_Y, entropy_Y_d = self.compute_entropy_marginal_binning_gpu_parallel(Y, num_bins)

        IG = entropy_Y - entropy_X
        IG_d = entropy_Y_d - entropy_X_d
        return IG, entropy_Y_d
    

    def compute_cosine_importance(self, X, Y):
        cos_sim = F.cosine_similarity(X, Y, dim=-1)
        score = 1 - float(torch.mean(cos_sim).detach().cpu())
        return score, 0
    

    def renyi_entropy(self, tensor, alpha=4.0):
        batch_size, seq_length, embed_dim = tensor.shape
        
        # 数据标准化
        X = tensor.view(-1, embed_dim)
        X = (X - X.mean(dim=0)) / (X.std(dim=0) + 1e-10)
        
        # 计算距离并添加缩放
        pairwise_dist = torch.cdist(X, X)
        sigma = torch.median(pairwise_dist)
        kernel_matrix = torch.exp(-pairwise_dist ** 2 / (2 * sigma ** 2))
        kernel_matrix = torch.clamp(kernel_matrix, min=1e-10)
        
        if alpha == 1.0:
            row_sums = torch.sum(kernel_matrix, dim=1, keepdim=True)
            probs = kernel_matrix / (row_sums + 1e-10)
            log_probs = torch.log(probs + 1e-10)
            entropy = -torch.sum(probs * log_probs) / X.shape[0]  # 归一化
        else:
            kernel_mean = torch.mean(kernel_matrix ** (alpha - 1))
            kernel_mean = torch.clamp(kernel_mean, min=1e-10)
            entropy = 1 / (1 - alpha) * torch.log(kernel_mean)
        
        # 添加最终检查
        if torch.isnan(entropy):
            print("WARNING: NaN detected in final entropy!")
            return 0.0
        
        return entropy.item()


    def information_gain_renyi(self, X, Y):
        entropy_X = self.renyi_entropy(X)
        entropy_Y = self.renyi_entropy(Y)
        IG = entropy_Y - entropy_X
        return IG, 0


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value = None,
        drop_attentions = None,
        drop_layer = None,
        last_key_states= None,
        last_query_states= None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings = None,  # will become mandatory in v4.46
        **kwargs,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        if drop_layer:
            return (hidden_states,), 0, 0, 0, 0, 0
        
        residual = hidden_states

        orig_input = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        if drop_attentions:
            hidden_states = residual
        else:
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

            hidden_states = residual + hidden_states

        score_attn, _ = self.compute_information_gain_fixed(residual.detach(), hidden_states.detach())
        

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        score_layer, _ = self.compute_information_gain_fixed(orig_input, hidden_states)


        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs, score_layer, score_attn


class LlamaModelDrop(LlamaModel):
    # init the model 
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.drop_attentions = None
        self.layers = nn.ModuleList([LlamaDecoderLayerDrop(config, i) for i in range(config.num_hidden_layers)])
        self.layer_num = len(self.layers)

        self.norms = [0 for _ in range(self.config.num_hidden_layers - 1)]


    # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
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
        prefilling_state: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        # Changed to output attention 
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
            attention_mask=attention_mask, input_tensor=inputs_embeds, cache_position=cache_position, past_key_values=past_key_values, output_attentions=output_attentions
        )
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        drop_score_list = []
        drop_score_list_layer = []
        entropy_y_d_list = []

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            output_attentions = False

            drop_attention = False
            drop_layer = False

            layer_outputs, drop_score_layer, drop_score_attn = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                drop_attentions = drop_attention,
                drop_layer = drop_layer,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

            drop_score_list.append(drop_score_attn)
            drop_score_list_layer.append(drop_score_layer)

            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)


        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        return output, drop_score_list


    

class LlamaForCausalLMDrop(LlamaForCausalLM):
    # init the model 
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = LlamaModelDrop(config)

        self.current_drop_order = 0
        print(self.model.dtype)

    def is_prefilling_stage(self, input_ids):
        seq_len = input_ids.size(1)
        return seq_len > 1

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
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

        hidden_states = outputs[0]
        
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )




    def process_layers(self, dataloader):
        norms = [0 for _ in range(self.config.num_hidden_layers)]
        for batch in tqdm(dataloader):

            inputs = {}
            inputs["input_ids"] = batch[0].cuda()
            outputs = self.model(**inputs, output_attentions=False)
            
            attention_drop_score = outputs[-1]
            for i in range(len(norms)):
                norms[i] += attention_drop_score[i]

        norms = [norm / len(dataloader) for norm in norms]

        # get the first two layers and last layer norm to be inf 
        norms[0] = float('inf')
        norms[1] = float('inf')
        norms[-1] = float('inf')
        

        # 排序时先按 norm 排序，相同时按索引排序
        sorted_norms = sorted(enumerate(norms), key=lambda x: (x[1], x[0]))

        # 分配唯一排名
        ranks = {idx: rank for rank, (idx, _) in enumerate(sorted_norms)}

        # 转换为排名顺序
        drop_layers_order = [ranks[i] for i in range(len(norms))]

        self.model.drop_layers_order = drop_layers_order

        self.config.drop_layers_order = self.model.drop_layers_order

        drop_layers_order = "_".join([str(i) for i in drop_layers_order])
        print(drop_layers_order)
        return drop_layers_order

