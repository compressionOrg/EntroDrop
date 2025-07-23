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
    def __init__(self, config: LlamaConfig, layer_idx: int, metric: str = "mse_normalized_combo"):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttentionDrop(config=config, layer_idx=layer_idx)
        self.layer_idx = layer_idx
        self.metric = metric  # "l1", "l2", "kl_divergence", "mse", "taylor", or "cosine_similarity"

    def compute_rrc(self, h_prev: torch.Tensor, h_curr: torch.Tensor) -> float:
        """Computes the Relative Residual Contribution (RRC) using different metrics.

        Args:
            h_prev: The hidden state before the block.
            h_curr: The hidden state after the block.

        Returns:
            The RRC score as a float.
        """
        # Ensure tensors are detached and on CPU for calculation
        h_prev = h_prev.detach().to(torch.float32)
        h_curr = h_curr.detach().to(torch.float32)

        if self.metric == "l2":
            # Original L2 norm method: RRC = ||h_{l+1} - h_l||₂ / ||h_l||₂
            # Flatten tensors to handle multi-dimensional inputs
            h_prev_flat = h_prev.view(-1)
            h_curr_flat = h_curr.view(-1)
            update_norm = torch.linalg.norm(h_curr_flat - h_prev_flat, ord=2)
            input_norm = torch.linalg.norm(h_prev_flat, ord=2)
            rrc_score = update_norm / (input_norm + 1e-10)
        elif self.metric == "l1":
            # L1 norm method: RRC = ||h_{l+1} - h_l||₁ / ||h_l||₁
            # Flatten tensors to handle multi-dimensional inputs
            h_prev_flat = h_prev.view(-1)
            h_curr_flat = h_curr.view(-1)
            update_norm = torch.linalg.norm(h_curr_flat - h_prev_flat, ord=1)
            input_norm = torch.linalg.norm(h_prev_flat, ord=1)
            rrc_score = update_norm / (input_norm + 1e-10)
        elif self.metric == "kl_divergence":
            # KL divergence method
            rrc_score = self._compute_kl_divergence(h_prev, h_curr)
        # elif self.metric == "mse1":
            # Mean Squared Error method: MSE = mean((h_curr - h_prev)²) / mean(h_prev²)
            # Flatten tensors to handle multi-dimensional inputs
            # h_prev_flat = h_prev.view(-1)
            # h_curr_flat = h_curr.view(-1)
            # mse_diff = torch.mean((h_curr_flat - h_prev_flat) ** 2)
            # mse_input = torch.mean(h_prev_flat ** 2)
            # rrc_score = mse_diff / (mse_input + 1e-10)
        elif self.metric == "mse":
            h_prev_flat = h_prev.view(-1)
            h_curr_flat = h_curr.view(-1)
            mse = torch.mean((h_curr_flat - h_prev_flat) ** 2)
            cos_sim = F.cosine_similarity(h_prev_flat.unsqueeze(0), h_curr_flat.unsqueeze(0), dim=1)[0]
            rrc_score = mse + (1 - cos_sim)
        elif self.metric == "cos":
            h_prev_flat = h_prev.view(-1)
            h_curr_flat = h_curr.view(-1)
            cos_sim = F.cosine_similarity(h_prev_flat.unsqueeze(0), h_curr_flat.unsqueeze(0), dim=1)[0]
            rrc_score = 1 - cos_sim
        elif self.metric == "mse_normalized_combo":
            # Combination after Normalization: Normalize MSE and (1 - cos_sim) before combining
            h_prev_flat = h_prev.view(-1)
            h_curr_flat = h_curr.view(-1)
            
            # Calculate MSE and (1 - Cosine Similarity)
            mse = torch.mean((h_curr_flat - h_prev_flat) ** 2)
            cos_sim_term = 1 - F.cosine_similarity(h_prev_flat.unsqueeze(0), h_curr_flat.unsqueeze(0), dim=1)[0]
            
            # Normalize both terms
            # Sigmoid for MSE to scale it to (0, 1)
            mse_norm = torch.sigmoid(mse)
            # The cosine similarity term is already in [0, 2], scale it to [0, 1]
            cos_sim_term_norm = cos_sim_term / 2.0
            
            # Combine the normalized scores
            rrc_score = mse_norm + cos_sim_term_norm
        elif self.metric == "taylor":
            # Taylor expansion method: approximates the function change using first and second order terms
            # RRC = (||Δh||₂ + 0.5 * ||Δh||₂²/||h_prev||₂) / ||h_prev||₂
            # Flatten tensors to handle multi-dimensional inputs
            h_prev_flat = h_prev.view(-1)
            h_curr_flat = h_curr.view(-1)
            delta_h = h_curr_flat - h_prev_flat
            
            # First order term: ||Δh||₂
            first_order = torch.linalg.norm(delta_h, ord=2)
            
            # Second order term: 0.5 * ||Δh||₂²/||h_prev||₂
            h_prev_norm = torch.linalg.norm(h_prev_flat, ord=2)
            second_order = 0.5 * (first_order ** 2) / (h_prev_norm + 1e-10)
            
            # Combined Taylor approximation
            taylor_approx = first_order + second_order
            rrc_score = taylor_approx / (h_prev_norm + 1e-10)
        elif self.metric == "spectral_norm":
            # Spectral Norm method: RRC = spectral_norm(h_curr - h_prev) / spectral_norm(h_prev)
            delta_h = h_curr - h_prev
            delta_flat = delta_h.view(delta_h.shape[0] * delta_h.shape[1], -1)
            h_prev_flat = h_prev.view(h_prev.shape[0] * h_prev.shape[1], -1)
            spectral_norm_delta = torch.linalg.svdvals(delta_flat).max()
            spectral_norm_prev = torch.linalg.svdvals(h_prev_flat).max()
            rrc_score = spectral_norm_delta / (spectral_norm_prev + 1e-10)
        elif self.metric == "mutual_information":
            # Mutual Information based: Use conditional entropy H(curr | prev) as unique contribution
            h_prev_flat = h_prev.view(-1).cpu().numpy()
            h_curr_flat = h_curr.view(-1).cpu().numpy()
            if len(h_prev_flat) < 2 or len(h_curr_flat) < 2:
                return 0.0  # Not enough samples
            # Entropy of prev
            kde_prev = gaussian_kde(h_prev_flat)
            h_prev_ent = -np.mean(kde_prev.logpdf(h_prev_flat))
            # Joint entropy
            joint_data = np.vstack([h_prev_flat, h_curr_flat])
            kde_joint = gaussian_kde(joint_data)
            h_joint = -np.mean(kde_joint.logpdf(joint_data))
            # Conditional entropy H(curr|prev) = H(joint) - H(prev)
            rrc_score = h_joint - h_prev_ent
        elif self.metric == "wasserstein":
            # Wasserstein Distance based: Quantify the "work" to move from prev to curr distribution
            from scipy.stats import wasserstein_distance
            h_prev_flat = h_prev.view(-1).cpu().numpy()
            h_curr_flat = h_curr.view(-1).cpu().numpy()
            rrc_score = wasserstein_distance(h_prev_flat, h_curr_flat)
        else:
            raise ValueError(f"Unknown metric: {self.metric}. Supported metrics: 'l1', 'l2', 'kl_divergence', 'mse', 'mse_normalized_combo', 'taylor', 'cos', 'spectral_norm', 'mutual_information', 'wasserstein'")

        return rrc_score.item() if isinstance(rrc_score, torch.Tensor) else rrc_score

    def _compute_kl_divergence(self, h_prev: torch.Tensor, h_curr: torch.Tensor) -> float:
        """Computes KL divergence between distributions of h_prev and h_curr.
        
        Args:
            h_prev: The hidden state before the block.
            h_curr: The hidden state after the block.
            
        Returns:
            KL divergence score as a float.
        """
        # Flatten tensors to compute distributions
        h_prev_flat = h_prev.view(-1)
        h_curr_flat = h_curr.view(-1)
        
        # Convert to probability distributions using softmax
        # Add small epsilon to avoid log(0)
        eps = 1e-10
        p_prev = F.softmax(h_prev_flat, dim=0) + eps
        p_curr = F.softmax(h_curr_flat, dim=0) + eps
        
        # Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
        kl_div = torch.sum(p_prev * torch.log(p_prev / p_curr))
        
        return kl_div.item()

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
        
        orig_input = hidden_states
        residual = hidden_states

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

        # Calculate RRC for the Attention block (default: skip calculation)
        # rrc_attn = self.compute_rrc(residual, hidden_states)
        rrc_attn = 0  # Default to 0, not calculating attention RRC

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Calculate RRC for the entire Layer (default: skip calculation)
        rrc_layer = self.compute_rrc(orig_input, hidden_states)
        # rrc_layer = 0  # Default to 0, not calculating layer RRC

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs, rrc_layer, rrc_attn


class LlamaModelDrop(LlamaModel):
    # init the model 
    def __init__(self, config, metric: str = "mse_normalized_combo"):
        super().__init__(config)
        self.config = config
        self.drop_attentions = None
        self.metric = metric
        self.layers = nn.ModuleList([LlamaDecoderLayerDrop(config, i, metric) for i in range(config.num_hidden_layers)])
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

        rrc_scores_attn = []
        rrc_scores_layer = []

        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            output_attentions = False

            drop_attention = False
            drop_layer = False

            layer_outputs, rrc_layer, rrc_attn = decoder_layer(
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

            rrc_scores_attn.append(rrc_attn)
            rrc_scores_layer.append(rrc_layer)

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

        return output, rrc_scores_layer, rrc_scores_attn


    

class LlamaForCausalLMDrop(LlamaForCausalLM):
    # init the model 
    def __init__(self, config, metric: str = "mse_normalized_combo"):
        super().__init__(config)
        self.config = config
        self.metric = metric
        self.model = LlamaModelDrop(config, metric)

        self.current_drop_order = 0
        print(f"Model dtype: {self.model.dtype}, Using metric: {self.metric}")

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

        # model_outputs will be a tuple: (last_hidden_state, rrc_scores_layer, rrc_scores_attn)
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

        hidden_states = model_outputs[0]
        
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        # The final output tuple that is returned by the model
        output_tuple = (logits,) + model_outputs[1:]

        if not return_dict:
            return (loss,) + output_tuple if loss is not None else output_tuple

        # The BaseModelOutputWithPast object from the model forward pass
        base_model_output = model_outputs[0]
        if not isinstance(base_model_output, BaseModelOutputWithPast):
             # if return_dict=False, model_outputs[0] is just hidden_states
             base_model_output = BaseModelOutputWithPast(
                last_hidden_state=model_outputs[0],
                past_key_values=model_outputs[1] if len(model_outputs) > 1 else None,
                hidden_states=model_outputs[2] if len(model_outputs) > 2 else None,
                attentions=model_outputs[3] if len(model_outputs) > 3 else None,
            )


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=base_model_output.past_key_values,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions,
        )




    def process_layers(self, dataloader):
        rrc_scores_layer = [0 for _ in range(self.config.num_hidden_layers)]
        rrc_scores_attn = [0 for _ in range(self.config.num_hidden_layers)]

        for batch in tqdm(dataloader):

            inputs = {}
            inputs["input_ids"] = batch[0].cuda()
            # The model forward now returns a tuple where the last two elements are our RRC scores
            outputs = self.model(**inputs, output_attentions=False)
            
            layer_scores = outputs[-2]
            attn_scores = outputs[-1]

            for i in range(len(rrc_scores_layer)):
                rrc_scores_layer[i] += layer_scores[i]
                rrc_scores_attn[i] += attn_scores[i]

        rrc_scores_layer = [score / len(dataloader) for score in rrc_scores_layer]
        rrc_scores_attn = [score / len(dataloader) for score in rrc_scores_attn]

        print("Layer RRC Scores:", rrc_scores_layer)
        print("Attention RRC Scores:", rrc_scores_attn)

        # get the first two layers and last layer norm to be inf 
        # We will use the layer RRC scores for pruning decisions
        scores_for_ranking = rrc_scores_layer
        scores_for_ranking[0] = float('inf')
        scores_for_ranking[1] = float('inf')
        scores_for_ranking[-1] = float('inf')
        

        # 排序时先按 norm 排序，相同时按索引排序
        sorted_scores = sorted(enumerate(scores_for_ranking), key=lambda x: (x[1], x[0]))

        # 分配唯一排名
        ranks = {idx: rank for rank, (idx, _) in enumerate(sorted_scores)}

        # 转换为排名顺序
        drop_layers_order = [ranks[i] for i in range(len(scores_for_ranking))]

        self.model.drop_layers_order = drop_layers_order

        self.config.drop_layers_order = self.model.drop_layers_order

        drop_layers_order_str = ",".join([str(i) for i in drop_layers_order])
        print(drop_layers_order_str)
        return drop_layers_order_str

