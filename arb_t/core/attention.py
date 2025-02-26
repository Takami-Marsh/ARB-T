import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for processing sequences."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        
        scaling = float(self.head_dim) ** -0.5
        
        # Linear projections and reshape
        q = self.q_proj(query) * scaling
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to (batch, head, seq_len, head_dim)
        q = self._shape(q, tgt_len, batch_size)
        k = self._shape(k, src_len, batch_size)
        v = self._shape(v, src_len, batch_size)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        
        # Apply masks if provided
        if attention_mask is not None:
            attn_weights = attn_weights.masked_fill(attention_mask, float('-inf'))
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
            
        # Normalize attention weights
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)
        
        # Compute output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, tgt_len, embed_dim
        )
        
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights

class SelfAttention(MultiHeadAttention):
    """Self-attention layer."""
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(x, x, x, key_padding_mask, attention_mask)

class CrossAttention(MultiHeadAttention):
    """Cross-attention layer for attending to external information."""
    
    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().forward(query, context, context, key_padding_mask, attention_mask)

class PositionalEncoding(nn.Module):
    """Inject information about relative or absolute position of tokens in sequence."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence length dimension
        return (x + self.pe[:, :x.size(1)]).squeeze(1) if x.size(1) == 1 else x + self.pe[:, :x.size(1)]
