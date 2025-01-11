import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)


class CrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, self.head_dim * num_heads)
        self.k_proj = nn.Linear(d_model, self.head_dim * num_heads)
        self.v_proj = nn.Linear(d_model, self.head_dim * num_heads)

        self.out_proj = nn.Linear(self.head_dim * num_heads, d_model)
        self.norm_q = RMSNorm(d_model)
        self.norm_kv = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, kv: torch.Tensor):
        bsz, seq_len, _ = x.shape
        _, kv_seq_len, _ = kv.shape

        x = self.norm_q(x)
        kv = self.norm_kv(kv)

        xq = self.q_proj(x)
        xk = self.k_proj(kv)
        xv = self.v_proj(kv)

        output_shape = xq.shape
        xq = xq.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, kv_seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        atten = torch.matmul(xq, xk.transpose(-2, -1)) * self.scale
        atten = F.softmax(atten, dim=-1)
        atten = F.dropout(atten, self.dropout)

        output = torch.matmul(atten, xv)
        output = output.transpose(1, 2).contiguous().view(output_shape)
        output = self.out_proj(output)
        return x + output


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_output: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()


class LocalEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_output: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_seq_len: int,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_output = d_output
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
