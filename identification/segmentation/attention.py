import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention for cross-modal fusion"""

    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.scale = 1.0 / math.sqrt(self.d_k)

    
    def forward(self, query, key, value):

        batch_size = query.size(0)

        # Linear projections

        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1,2)

        # Attention computation

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to V

        context = torch.matmul(attention_weights, V)

        # Concatenate heads and project

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.w_o(context)




