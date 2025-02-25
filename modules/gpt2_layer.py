from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention
from modules.lora import LoRALinear


class GPT2Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Multi-head attention.
        self.self_attention = CausalSelfAttention(config)
        # Add-norm for multi-head attention.
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        # Feed forward.
        self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.interm_af = F.gelu
        # Add-norm for feed forward.
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.out_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

        # If using LoRA, create additional LoRA layers
        self.use_lora = config.use_lora
        if config.use_lora:
            self.lora_attention_dense = LoRALinear(config.hidden_size, config.hidden_size,
                                                 rank=config.lora_rank, alpha=config.lora_alpha)
            self.lora_interm_dense = LoRALinear(config.hidden_size, config.intermediate_size,
                                               rank=config.lora_rank, alpha=config.lora_alpha)
            self.lora_out_dense = LoRALinear(config.intermediate_size, config.hidden_size,
                                           rank=config.lora_rank, alpha=config.lora_alpha)

            # Freeze the original weights
            self.attention_dense.weight.requires_grad = False
            self.attention_dense.bias.requires_grad = False
            self.interm_dense.weight.requires_grad = False
            self.interm_dense.bias.requires_grad = False
            self.out_dense.weight.requires_grad = False
            self.out_dense.bias.requires_grad = False

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
        attention = self.self_attention(
            self.attention_layer_norm(hidden_states),
            attention_mask)

        if self.use_lora:
            out = self.add(hidden_states, attention,
                        self.lora_attention_dense, self.attention_dropout)
        else:
            out = self.add(hidden_states, attention,
                        self.attention_dense, self.attention_dropout)

        mlp_input = self.out_layer_norm(out)
        if self.use_lora:
            mlp = self.interm_af(self.lora_interm_dense(mlp_input))
            out = self.add(out, mlp, self.lora_out_dense, self.out_dropout)
        else:
            mlp = self.interm_af(self.interm_dense(mlp_input))
            out = self.add(out, mlp, self.out_dense, self.out_dropout)

        return out
