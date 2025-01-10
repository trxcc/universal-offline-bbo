import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class ByteTransformer(nn.Module):
    def __init__(
        self, d_model=256, nhead=4, num_layers=6, dim_feedforward=512, dropout=0.1
    ):
        super().__init__()

        # 256种可能的byte值的embedding
        self.embedding = nn.Embedding(258, d_model)

        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, 512, d_model))

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
