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
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, self.head_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim * num_heads, bias=False)

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

class Attention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float,
    ):
        super().__init__()



class TransformerBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            dropout: float,
    ):
        super().__init__()

        

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)





class LocalEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_output: int,
        num_layers: int,    
        num_heads: int,
        cross_attn_nheads: int,
        dropout: float,
        max_seq_len: int,
        max_patch_len: int=256,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_output = d_output
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.cross_attn_nheads = cross_attn_nheads
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.max_patch_len = max_patch_len
        self.tok_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model * max_patch_len, d_output)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=num_heads, 
                dropout=dropout,
                dim_feedforward=d_model * 4,
                norm_first=True,
                batch_first=True,
                ) for _ in range(num_layers)
        ])
        self.cross_attn_layers = nn.ModuleList([
            CrossAttention(d_model, cross_attn_nheads, dropout) for _ in range(num_layers)
        ])
    
    def apply_embedding(self, tokens):
        return self.tok_embeddings(tokens)
    
    def forward(self, tokens: torch.Tensor, patch_ids: torch.Tensor):
        text_embeds = self.apply_embedding(tokens)
        # assert 0
        text_embeds = F.dropout(text_embeds, self.dropout)
        patch_embeds = self.get_patch_embeds(patch_ids, text_embeds, self.max_patch_len)
        try:    
            bsz, _, d_model = text_embeds.shape
            for i, layer in enumerate(self.layers):
                text_embeds = layer(text_embeds)
            patch_embeds = self.cross_attn_layers[i](patch_embeds, text_embeds)
        except Exception as e:
            print(e)
            print("forward error2")
            raise e

        return self.out_proj(patch_embeds.view(bsz, -1))
    
    def get_patch_embeds(self, patch_ids: torch.Tensor, text_embeds: torch.Tensor, patch_max_len: int):
        bsz, _, d_model = text_embeds.shape
        # print(patch_ids.shape)
        # print(text_embeds.shape)
        patch_embeds = torch.zeros(bsz, patch_max_len, d_model, device=text_embeds.device)
        index = patch_ids.unsqueeze(-1)
        # print(index.shape)
        # print(index.expand(-1, -1, d_model).shape)
        # print(patch_ids.max())
        # assert 0
        patch_embeds.scatter_reduce_(dim=1, index=index.expand(-1, -1, d_model), src=text_embeds, reduce="mean")
        # assert 0
        # print(patch_embeds.shape)
        return patch_embeds
if __name__ == "__main__":
    model = LocalEncoder(100, 128, 128, 2, 8, 8, 0.1, 1024)
    tokens = torch.randint(0, 100, (3, 1024))
    print(tokens.shape)
    patch_ids = torch.randint(0, 256, (3, 1024))
    print(patch_ids.shape)
    print(model(tokens, patch_ids).shape)