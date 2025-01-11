import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteTransformer(nn.Module):
    def __init__(
        self, d_model=256, nhead=4, num_layers=6, dim_feedforward=512, dropout=0.1
    ):
        super().__init__()

        # Embeddings for 256 possible bytes
        self.embedding = nn.Embedding(258, d_model)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 2048, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection to byte probabilities
        self.fc_out = nn.Linear(d_model, 258)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        seq_len = x.size(1)

        # Get embeddings
        x = self.embedding(x)  # [batch_size, seq_len, d_model]

        # Add positional encoding
        x = x + self.pos_encoder[:, :seq_len, :]

        # Transformer expects [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)

        # Pass through transformer
        x = self.transformer(x)

        # Back to [batch_size, seq_len, d_model]
        x = x.transpose(0, 1)

        # Project to byte probabilities
        logits = self.fc_out(x)  # [batch_size, seq_len, 258]

        return logits

    def get_single_logits(self, text_tokens: torch.Tensor) -> torch.Tensor:
        text_tokens = text_tokens.unsqueeze(0)
        logits = self(text_tokens)
        logits = logits.reshape(-1, logits.shape[-1])
        return logits

    def load_from_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        checkpoint = self.on_load_checkpoint(checkpoint)
        self.load_state_dict(checkpoint["state_dict"])

    def on_load_checkpoint(self, checkpoint: dict):
        keys_list = list(checkpoint["state_dict"].keys())
        for key in keys_list:
            if "orig_mod." in key:
                deal_key = key.replace("model._orig_mod.", "")
                checkpoint["state_dict"][deal_key] = checkpoint["state_dict"][key]
                del checkpoint["state_dict"][key]
        return checkpoint
