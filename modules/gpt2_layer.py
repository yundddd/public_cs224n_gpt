from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention
from modules.lora import LoRALinear


class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_lora = config.use_lora
        
        # Multi-head attention components
        self.self_attention = CausalSelfAttention(config)
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        
        # Conditional layer creation for attention output
        self.attention_dense = (
            LoRALinear(config.hidden_size, config.hidden_size, 
                     rank=config.lora_rank, alpha=config.lora_alpha)
            if self.use_lora 
            else nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)

        # Feed-forward components
        self.interm_dense = (
            LoRALinear(config.hidden_size, config.intermediate_size,
                     rank=config.lora_rank, alpha=config.lora_alpha) 
            if self.use_lora
            else nn.Linear(config.hidden_size, config.intermediate_size)
        )
        self.interm_af = F.gelu
        
        self.out_dense = (
            LoRALinear(config.intermediate_size, config.hidden_size,
                     rank=config.lora_rank, alpha=config.lora_alpha)
            if self.use_lora
            else nn.Linear(config.intermediate_size, config.hidden_size)
        )
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add(self, input, output, dense_layer, dropout):
        """
        TODO: Implement this helper method for the forward function.
          - This function is applied after the multi-head attention layer as well as after the feed forward layer.
          - GPT-2 layer applies dropout to the transformed output of each sub-layer,
            before it is added to the sub-layer input. WE DO NOT APPLY THE LAYER NORM
            IN THIS FUNCTION.
        """
        output = dropout(dense_layer(output))
        return input + output

    def forward(self, hidden_states, attention_mask):
        """
        TODO: Implement the forward pass. Some key points to consider:
               - A multi-head attention layer (CausalSelfAttention) that computes self-attention based on masked inputs.
               - Layer normalization applied *before* the attention layer and feed-forward layer.
               - Apply dropout, residual connection, and layer normalization according to the plot in the assignment. (Use self.add)
               - A feed-forward layer that applies transformations to further refine the hidden states.
        """
        # Attention path
        attn_norm = self.attention_layer_norm(hidden_states)
        attention = self.self_attention(attn_norm, attention_mask)
        out = self.add(hidden_states, attention, 
                      self.attention_dense, self.attention_dropout)

        # Feed-forward path
        mlp_norm = self.out_layer_norm(out)
        mlp = self.interm_af(self.interm_dense(mlp_norm))
        out = self.add(out, mlp, self.out_dense, self.out_dropout)

        return out
