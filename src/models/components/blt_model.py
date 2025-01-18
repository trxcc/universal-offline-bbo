from typing import Optional, Tuple

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


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):

    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


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
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, self.head_dim * num_heads, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim * num_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.head_dim * num_heads, bias=False)

        self.out_proj = nn.Linear(self.head_dim * num_heads, d_model)

    def forward(self, x: torch.Tensor, freq_cis: torch.Tensor):
        bsz, seq_len, dim = x.shape
        x_q = self.q_proj(x)
        x_k = self.k_proj(x)
        x_v = self.v_proj(x)

        x_q = x_q.view(bsz, seq_len, self.num_heads, self.head_dim)
        x_k = x_k.view(bsz, seq_len, self.num_heads, self.head_dim)
        x_v = x_v.view(bsz, seq_len, self.num_heads, self.head_dim)

        x_q, x_k = apply_rotary_emb(x_q, x_k, seq_dim=1, freqs_cis=freq_cis[0:seq_len])

        x_q = x_q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        x_k = x_k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        x_v = x_v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        atten = torch.matmul(x_q, x_k.transpose(-2, -1)) * self.scale
        atten = F.softmax(atten, dim=-1)
        atten = F.dropout(atten, self.dropout)

        output = torch.matmul(atten, x_v)
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        output = self.out_proj(output)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.d_model ** (-0.5))

        for w in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.out_proj.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
    ):
        super().__init__()

        hidden_dim = int(8 * d_model / 3)

        self.d_model = d_model
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            d_model,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            d_model,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            d_model,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.d_model ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.attention = Attention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model)
        self.attention_norm = RMSNorm(d_model)
        self.feed_forward_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, freq_cis: torch.Tensor):
        h = x + self.attention(self.attention_norm(x), freq_cis)
        output = h + self.feed_forward(self.feed_forward_norm(h))
        return output

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.feed_forward.reset_parameters(init_std, factor)


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
        max_patch_len: int = 256,
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
        self.rotary_embedding = RotaryEmbedding(
            theta=10000.0, head_dim=d_model // num_heads
        )
        self.norm = nn.LayerNorm(d_model)
        self.patch_norm = nn.LayerNorm(d_model)
        # self.out_proj = nn.Linear(d_model * max_patch_len, d_output)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )

        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttention(d_model, cross_attn_nheads, dropout)
                for _ in range(num_layers)
            ]
        )

    def apply_embedding(self, tokens):
        return self.tok_embeddings(tokens)

    def forward(self, tokens: torch.Tensor, patch_ids: torch.Tensor):
        bs, seqlen = tokens.shape
        text_embeds = self.apply_embedding(tokens)
        text_embeds = F.dropout(text_embeds, self.dropout)
        freq_cis = self.rotary_embedding(seqlen=seqlen)
        patch_embeds = self.get_patch_embeds(patch_ids, text_embeds, self.max_patch_len)
        try:
            bsz, _, d_model = text_embeds.shape
            for i, layer in enumerate(self.layers):
                text_embeds = layer(text_embeds, freq_cis)
                patch_embeds_cross = self.cross_attn_layers[i](
                    patch_embeds, text_embeds
                )
                patch_embeds = patch_embeds_cross + patch_embeds
        except Exception as e:
            print(e)
            print("forward error2")
            raise e

        # return self.out_proj(patch_embeds.view(bsz, -1))
        return patch_embeds

    def get_patch_embeds(
        self, patch_ids: torch.Tensor, text_embeds: torch.Tensor, patch_max_len: int
    ):
        bsz, _, d_model = text_embeds.shape
        # print(patch_ids.shape)
        # print(text_embeds.shape)
        patch_embeds = torch.zeros(
            bsz, patch_max_len, d_model, device=text_embeds.device
        )
        index = patch_ids.unsqueeze(-1)
        # print(index.shape)
        # print(index.expand(-1, -1, d_model).shape)
        # print(patch_ids.max())
        # assert 0
        patch_embeds.scatter_reduce_(
            dim=1, index=index.expand(-1, -1, d_model), src=text_embeds, reduce="mean"
        )
        # assert 0
        # print(patch_embeds.shape)
        return patch_embeds

    def init_weights(self, init_std=None, factor=1.0):
        for layer in self.layers:
            layer.init_weights(init_std, factor)


class GlobalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.rotary_embedding = RotaryEmbedding(
            theta=10000.0, head_dim=d_model // num_heads
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        bs, seqlen, _ = x.shape
        freq_cis = self.rotary_embedding(seqlen=seqlen)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, freq_cis)
        return x

    def init_weights(self, init_std=None, factor=1.0):
        for layer in self.layers:
            layer.init_weights(init_std, factor)


class LocalDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        num_cross_attn_heads: int,
        dropout: float,
        vocab_size: int = 258,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, dropout) for _ in range(num_layers)]
        )
        self.cross_attn_layers = nn.ModuleList(
            [
                CrossAttention(d_model, num_cross_attn_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.rotary_embedding = RotaryEmbedding(
            theta=10000.0, head_dim=d_model // num_heads
        )
        self.dropout = dropout
        self.norm = RMSNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, token_embeds: torch.Tensor, patch_embeds: torch.Tensor):
        bs, seqlen, _ = token_embeds.shape
        freq_cis = self.rotary_embedding(seqlen=seqlen)
        h = token_embeds
        h = F.dropout(h, self.dropout)
        for i, layer in enumerate(self.layers):
            h_cross = self.cross_attn_layers[i](patch_embeds, h)
            h = h_cross + h
            h = layer(h, freq_cis)
        h_pred = self.norm(h)
        h_pred = F.dropout(h_pred, self.dropout)
        h_pred = self.out_proj(h_pred)
        return h_pred.float()

    def init_weights(self, init_std=None, factor=1.0):
        for layer in self.layers:
            layer.init_weights(init_std, factor)


class BLTEmbedder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_output: int,
        num_encoder_layers: int,
        num_embedder_layers: int,
        num_heads: int,
        dropout: float,
        cross_attn_nheads: int,
        max_seq_len: int = 1024,
        max_patch_len: int = 256,
    ):
        super().__init__()
        self.local_encoder = LocalEncoder(
            vocab_size,
            d_model,
            d_output,
            num_encoder_layers,
            num_heads,
            cross_attn_nheads,
            dropout,
            max_seq_len,
            max_patch_len,
        )
        self.global_transformer = GlobalTransformer(
            d_model, num_embedder_layers, num_heads, dropout, max_seq_len
        )
        self.out_proj = nn.Linear(d_model * max_patch_len, d_output)

    def forward(self, x: torch.Tensor, patch_ids: torch.Tensor):
        bsz, seqlen = x.shape
        x = self.local_encoder(x, patch_ids)
        x = self.global_transformer(x)
        return self.out_proj(x.view(bsz, -1))

    def init_weights(self, init_std=None, factor=1.0):
        self.local_encoder.init_weights(init_std, factor)
        self.global_transformer.init_weights(init_std, factor)


class BLT(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_output: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_embedder_layers: int,
        num_heads: int,
        dropout: float,
        cross_attn_nheads: int,
        max_seq_len: int = 1024,
        max_patch_len: int = 256,
        vocab_size: int = 258,
    ):
        super().__init__()
        self.embedder = BLTEmbedder(
            d_model,
            d_output,
            num_encoder_layers,
            num_embedder_layers,
            num_heads,
            dropout,
            cross_attn_nheads,
            max_seq_len,
            max_patch_len,
        )
        self.decoder = LocalDecoder(
            d_model,
            num_decoder_layers,
            num_heads,
            cross_attn_nheads,
            dropout,
            vocab_size,
        )

    def forward(self, x: torch.Tensor, patch_ids: torch.Tensor):
        h, patch_embeds = self.embedder(x, patch_ids)
        return self.decoder(h, patch_embeds)

    def init_weights(self, init_std=None, factor=1.0):
        self.embedder.init_weights(init_std, factor)
        self.decoder.init_weights(init_std, factor)


if __name__ == "__main__":
    model = LocalEncoder(100, 128, 128, 2, 8, 8, 0.1, 1024)
    tokens = torch.randint(0, 100, (3, 1024))
    print(tokens.shape)
    patch_ids = torch.randint(0, 256, (3, 1024))
    print(patch_ids.shape)
    print(model(tokens, patch_ids).shape)
