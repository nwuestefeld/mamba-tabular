import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .normalization_layers import (
    RMSNorm,
    LayerNorm,
    LearnableLayerScaling,
    BatchNorm,
    InstanceNorm,
    GroupNorm,
)
from ..arch_utils.mamba_arch import Mamba

class Jamba(nn.Module):
    """Jamba model composed of multiple JambaBlocks.

    Attributes:
        config (MambaConfig): Configuration object for the Mamba model.
        layers (nn.ModuleList): List of JambaBlocks constituting the model.
    """

    def __init__(
        self,
        d_model=32,
        n_layers=8,
        expand_factor=2,
        bias=False,
        d_conv=8,
        conv_bias=True,
        dropout=0.01,
        dt_rank="auto",
        d_state=16,
        dt_scale=1.0,
        dt_init="random",
        dt_max=0.1,
        dt_min=1e-03,
        dt_init_floor=1e-04,
        norm=RMSNorm,
        activation=F.silu,
        bidirectional=False,
        use_learnable_interaction=False,
        layer_norm_eps=1e-05,
        AD_weight_decay=False,
        BC_layer_norm=True,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ResidualJambaBlock(
                    d_model,
                    expand_factor,
                    bias,
                    d_conv,
                    conv_bias,
                    dropout,
                    dt_rank,
                    d_state,
                    dt_scale,
                    dt_init,
                    dt_max,
                    dt_min,
                    dt_init_floor,
                    norm,
                    activation,
                    bidirectional,
                    use_learnable_interaction,
                    layer_norm_eps,
                    AD_weight_decay,
                    BC_layer_norm,

                    ##hier muss noch was MoE config 
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
    
class ResidualJambaBlock(nn.Module):
    """Residual block composed of a JambaBlock and a normalization layer.

    Attributes:
        layers (MambaBlock): MambaBlock layers.
        norm (RMSNorm): Normalization layer.
    """

    def __init__(
        self,
        d_model=32,
        expand_factor=2,
        bias=False,
        d_conv=16,
        conv_bias=True,
        dropout=0.01,
        dt_rank="auto",
        d_state=32,
        dt_scale=1.0,
        dt_init="random",
        dt_max=0.1,
        dt_min=1e-03,
        dt_init_floor=1e-04,
        norm=RMSNorm,
        activation=F.silu,
        bidirectional=False,
        use_learnable_interaction=False,
        layer_norm_eps=1e-05,
        AD_weight_decay=False,
        BC_layer_norm=False,
    ):
        super().__init__()

        VALID_NORMALIZATION_LAYERS = {
            "RMSNorm": RMSNorm,
            "LayerNorm": LayerNorm,
            "LearnableLayerScaling": LearnableLayerScaling,
            "BatchNorm": BatchNorm,
            "InstanceNorm": InstanceNorm,
            "GroupNorm": GroupNorm,
        }

        # Check if the provided normalization layer is valid
        if isinstance(norm, type) and norm.__name__ not in VALID_NORMALIZATION_LAYERS:
            raise ValueError(
                f"Invalid normalization layer: {norm.__name__}. "
                f"Valid options are: {', '.join(VALID_NORMALIZATION_LAYERS.keys())}"
            )
        elif isinstance(norm, str) and norm not in self.VALID_NORMALIZATION_LAYERS:
            raise ValueError(
                f"Invalid normalization layer: {norm}. "
                f"Valid options are: {', '.join(VALID_NORMALIZATION_LAYERS.keys())}"
            )

        if dt_rank == "auto":
            dt_rank = math.ceil(d_model / 16)


#TODO: Parameter anpassen
# need attention related parameters: heads, key_value heads, attn_dropout
# need MoE stuff: num_experts, num_experts_per_tok
# need Structred attention stuff: attn_layer_offset, attn_layer_period, expert_layer_offset, expert_layer_period
# need Mamba stuff: mamba_config
        self.layers = JambaBlock(
            d_model=d_model,
            expand_factor=expand_factor,
            bias=bias,
            d_conv=d_conv,
            conv_bias=conv_bias,
            dropout=dropout,
            dt_rank=dt_rank,
            d_state=d_state,
            dt_scale=dt_scale,
            dt_init=dt_init,
            dt_max=dt_max,
            dt_min=dt_min,
            dt_init_floor=dt_init_floor,
            activation=activation,
            bidirectional=bidirectional,
            use_learnable_interaction=use_learnable_interaction,
            layer_norm_eps=layer_norm_eps,
            AD_weight_decay=AD_weight_decay,
            BC_layer_norm=BC_layer_norm,
        )
        self.norm = norm(d_model, eps=layer_norm_eps)

    def forward(self, x):
        output = self.layers(self.norm(x)) + x
        return output
    




class JambaBlock(nn.Module):
    def __init__(self, config = Jambular_config):
        super().__init__()
        self.config = config    
        decoder_layers = []



#### Layer Anordnung
## AttentionLayer: GQA + MOE
## MambaLayer: Mamba + MOE 
        for i in range(config.n_layers):
             is_attn = True if (i - self.config.attn_layer_offset) % self.config.attn_layer_period == 0 else False
             is_expert = True if (i - self.config.expert_layer_offset) % self.config.expert_layer_period == 0 else False

             num_experts = self.config.num_experts if is_expert else 1

             if is_attn:
                decoder_layers.append(AttentionLayer(config, num_experts=num_experts))
             else:
                decoder_layers.append(MambaLayer(config, num_experts=num_experts))
        self.layers = nn.ModuleList(decoder_layers)

    def forward(self, x):
         # x: (B, L, D)

        # logits: (B, L, D)
        # router_logits : (B*L, n_experts)
         router_logits = []

         for decoder_layer in self.layers:
            layer_output, _ = decoder_layer(x)
            x = layer_output[0]
            router_logits.append(layer_output[1])

         return x, router_logits
    
    def step(self, x, caches):
        # x: (B, L, D)

        # logits: (B, L, D)
        # caches

        for i, decoder_layer in enumerate(self.layers):
            layer_output, caches[i] = decoder_layer(x, caches[i])
            x = layer_output[0]

        return x, caches
    
# Attention layer for jamba block
# In Paper: Attention-MoE Layer: Norm+Attention+Norm+MoE
# Warum???
class AttentionLayer(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int):
        super().__init__()

        self.self_attn = AttentionSDPA(config)

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, cache = None):
        # x: (B, L, D)

        # outputs: (B, L, D)
        
        #norm
        # attention
        residual = x
        x = self.input_layernorm(x)
        x, cache = self.self_attn(x, cache)
        x = residual + x

        # FFN
        #norm
        #moe
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)
        return outputs, cache

    def get_empty_cache(self, batch_size, device):
        return (None, None)


#dot product attention
class AttentionSDPA(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.config = config

        self.hidden_size = config.d_model
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, x, cache = None):
        # x: (B, L, D)

        # attn_output: (B, L, D)

        B, L, _ = x.size()

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = queries.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        values = values.view(B, L, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # kv cache implementation
        if cache is not None:
            past_keys, past_values = cache
            
            # not first in the sequence
            if past_keys is not None:
                keys = torch.cat([past_keys, keys], dim=2)
                values = torch.cat([past_values, values], dim=2)
            
            cache = (keys, values) # prepare cache for next token

        # GQA related
        keys = repeat_kv(keys, self.num_key_value_groups)
        values = repeat_kv(values, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(queries, keys, values,
                                                                       dropout_p=self.attention_dropout if self.training else 0.0,
                                                                       is_causal=(cache is None))
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, L, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, cache


#Main Mamba Layer: In Paper: Mamba-MoE Layer: Norm+Mamba+Norm+MoE 
class MambaLayer(nn.Module):
    def __init__(self, config: JambaLMConfig, num_experts: int):
        super().__init__()

        self.config = config

        self.mamba = MambaBlock(config=config.mamba_config)

        num_experts_per_tok = config.num_experts_per_tok if num_experts > 1 else 1
        self.moe = SparseMoEBlock(config, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok)
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

    def forward(self, x, cache = None):
        # x: (B, L, D)

        # outputs: (B, L, D)

        # mamba
        #norm+Mamba
        residual = x
        x = self.input_layernorm(x)
        if cache is None:
            x = self.mamba(x)
        else:
            x, cache = self.mamba.step(x.squeeze(1), cache)
            x = x.unsqueeze(1)
        x = residual + x

        # FFN
        #norm+MoE
        residual = x
        x = self.pre_moe_layernorm(x)
        x, router_logits = self.moe(x)
        x = residual + x

        outputs = (x, router_logits)

        return outputs, cache
    
    def get_empty_cache(self, batch_size, device):
        return (None, torch.zeros(batch_size, self.config.d_inner, self.config.d_conv-1, device=device))
    


###MOE Component
class SparseMoEBlock(nn.Module):
    def __init__(self, config: JambaConfig, num_experts: int, num_experts_per_tok: int):
        super().__init__()

        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok

        if num_experts > 1:
            self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        else:
            self.router = None

        self.experts = nn.ModuleList([MLP(config) for _ in range(self.num_experts)])
    def forward(self, x):
        # x: (B, L, D)

        # final_hidden_states: (B, L, D)
        # router_logits: (B*L, n_experts)

        #note : it is not clear why we work with shape (B*L, D) here.
        #I copied this code from the official jamba imple, and did not have time to think it through.
        
        batch_size, sequence_length, hidden_dim = x.shape

        # no routing
        if self.num_experts == 1:
            final_hidden_states = self.experts[0](x)
            router_logits = torch.ones(
                (batch_size * sequence_length, 1),
                device=x.device,
                dtype=x.dtype,
                requires_grad=x.requires_grad,
            )
            return final_hidden_states, router_logits

        # routing
        x = x.view(-1, hidden_dim) # (B*L, D)

        router_logits = self.router(x) # (B*L, n_experts)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights.to(x.dtype)

        final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=x.dtype, device=x.device)

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = x[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        
        return final_hidden_states, router_logits

###MLP for transformer Layer und Mamba Layer
class MLP(nn.Module):
    def __init__(self, config: JambaLMConfig):
        super().__init__()

        self.hidden_dim = config.d_model
        self.ffn_dim = config.mlp_size

        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, x):
        # x : (B, L, D)

        # y : (B, L, D)

        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    
def load_balancing_loss(router_logits, num_experts, num_experts_per_tok):
    # router_logits: list of router_logit, one per layer, each (B*D, n_experts)

    # moe_aux_loss : scalar

    router_logits = torch.cat([r for r in router_logits if r.shape[1] > 1], dim=0)

    routing_weights = torch.nn.functional.softmax(router_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # percentage of tokens routed to each experts
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # average probability of routing to these experts
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    moe_aux_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return moe_aux_loss * num_experts

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
