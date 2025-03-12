from modules.gpt2_layer import GPT2Layer
from modules.switch_expert import SwitchTransformerLayer


class GPT2MoELayer(GPT2Layer):
    def __init__(self, config):
        super().__init__(config)
        self.moe = SwitchTransformerLayer(config)

    def forward(self, hidden_states, attention_mask):
        attention = self.self_attention(
            self.attention_layer_norm(hidden_states),
            attention_mask)
        out = self.add(hidden_states, attention,
                       self.attention_dense, self.attention_dropout)
        mlp, aux_loss, expert_counts = self.moe(self.out_layer_norm(out))
        out = self.add(out, mlp, self.out_dense, self.out_dropout)
        return out, aux_loss, expert_counts
