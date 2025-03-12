import torch
import torch.nn as nn
import torch.nn.functional as F


class SwitchExpert(nn.Module):
    """ A single expert (Feed-Forward Network) in the Switch Transformer layer. """

    def __init__(self, d_model, d_ff, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_out)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SwitchRouter(nn.Module):
    """ Routing network for Switch Transformer using Top-1 gating. """

    def __init__(self, d_model, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts)

    # x is flattened list of token embeddings.
    # [batch_size * seq_len, d_model]
    def forward(self, x):
        # Compute routing logits [batch_size * seq_len, num_experts]
        logits = self.gate(x)
        # Convert to probabilities [batch_size * seq_len, num_experts]
        weights = F.softmax(logits, dim=-1)
        # Select highest probability expert [batch_size * seq_len]
        top1_expert = torch.argmax(weights, dim=-1)

        return top1_expert, weights


class SwitchTransformerLayer(nn.Module):
    """ Switch Transformer Layer with auxiliary loss for load balancing. """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([
            SwitchExpert(
                config.hidden_size, config.expert_hidden_size,
                config.intermediate_size)
            for _ in range(config.num_experts)])
        self.router = SwitchRouter(config.hidden_size, config.num_experts)
        self.dropout = nn.Dropout(0.1)
        self.aux_loss_weight = config.aux_loss_weight  # Weight for auxiliary loss

    # x is the output of attention layer [batch_size, seq_len, d_model]
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1, d_model)  # Flatten batch and sequence dimensions

        # Get routing decisions
        top1_expert, weights = self.router(x)

        # Compute auxiliary loss for load balancing
        expert_counts = torch.bincount(
            top1_expert, minlength=self.num_experts).to(
            x.dtype)  # Preserve dtype

        # Add small epsilon to avoid division by zero
        expert_probs = (expert_counts + 1e-10) / expert_counts.sum()
        uniform_dist = (torch.full_like(expert_probs, 1.0 / self.num_experts) + 1e-10)

        aux_loss = F.kl_div(expert_probs.log(), uniform_dist, reduction="batchmean")

        # Ensure aux_loss requires gradients
        aux_loss = aux_loss.clone().detach().requires_grad_(True)  # Track gradients

        # Create mask for expert selection
        output_size = list(x.size())
        output_size[-1] = self.config.intermediate_size
        expert_outputs = torch.zeros(
            torch.Size(output_size),
            device=x.device, dtype=x.dtype)

        for i in range(self.num_experts):
            expert_mask = top1_expert == i
            if expert_mask.any():
                expert_outputs[expert_mask] = self.experts[i](x[expert_mask])

        output = self.dropout(expert_outputs.view(
            batch_size, seq_len, self.config.intermediate_size))

        # Return output and properly tracked aux loss
        return output, self.aux_loss_weight * aux_loss, expert_counts
