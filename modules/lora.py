import math
import torch
from torch import nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=32):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scale = alpha / rank
        
        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze the original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        base_output = self.linear(x)
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base_output + lora_output

    # Add property getters/setters to expose the internal linear layer's attributes
    @property
    def weight(self):
        return self.linear.weight

    @weight.setter 
    def weight(self, value):
        self.linear.weight = value

    @property
    def bias(self):
        return self.linear.bias

    @bias.setter
    def bias(self, value):
        self.linear.bias = value