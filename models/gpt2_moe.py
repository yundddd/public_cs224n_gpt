from models.gpt2 import GPT2Model
from modules.gpt_with_moe_layer import GPT2MoELayer
from torch import nn, Tensor
import torch
from utils import get_extended_attention_mask


class GPT2MoEModel(GPT2Model):
    def __init__(self, config):
        super().__init__(config)
        # GPT-2 layers.
        self.gptmoe_layers = nn.ModuleList([GPT2MoELayer(config)
                                            for _ in range(config.num_moe_layers)])

    def encode(self, hidden_states, attention_mask):
        """
        hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]
        """
        extended_attention_mask: Tensor = get_extended_attention_mask(
            attention_mask,
            self.dtype)

        for _, layer_module in enumerate(self.gpt_layers):
            hidden_states = layer_module(hidden_states, extended_attention_mask)

        total_aux_loss = torch.tensor(0.0, device=hidden_states.device)
        expert_layer_count = torch.zeros(
            (len(self.gptmoe_layers), len(self.gptmoe_layers[0].moe.experts)),
            device=hidden_states.device, dtype=torch.int64)
        for i, layer_module in enumerate(self.gptmoe_layers):
            hidden_states, aux_loss, expert_counts = layer_module(
                hidden_states, extended_attention_mask)
            total_aux_loss += aux_loss
            expert_layer_count[i] = expert_counts

        return hidden_states, total_aux_loss, expert_layer_count

    def forward(self, input_ids, attention_mask):
        """
        input_ids: [batch_size, seq_len], seq_len is the max length of the batch
        attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
        """
        # Get the embedding for each input token.
        embedding_output = self.embed(input_ids=input_ids)

        # Feed to a transformer (a stack of GPTLayers).
        sequence_output, aux_loss, expert_layer_count = self.encode(
            embedding_output, attention_mask=attention_mask)
        sequence_output = self.final_layer_norm(sequence_output)

        # Get the hidden state of the final token.
        last_non_pad_idx = attention_mask.sum(dim=1) - 1  # Subtract 1 to get last index
        last_token = sequence_output[torch.arange(
            sequence_output.shape[0]), last_non_pad_idx]
        return {'last_hidden_state': sequence_output, 'last_token': last_token, 'aux_loss': aux_loss, 'expert_layer_count': expert_layer_count}
