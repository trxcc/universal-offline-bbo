import torch
import torch.nn as nn
from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Block, T5Stack


class CustomT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

    def forward(self, input_ids=None, attention_mask=None, emb_meta=None, **kwargs):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        all_hidden_states = () if self.config.output_hidden_states else None
        all_attentions = () if self.config.output_attentions else None

        if emb_meta is not None:
            emb_meta = emb_meta.unsqueeze(1).expand(-1, hidden_states.size(1), -1)

        for i, block in enumerate(self.block):
            hidden_states = block(
                hidden_states, attention_mask=attention_mask, **kwargs
            )[0]

            if emb_meta is not None:
                hidden_states = hidden_states + emb_meta

            if self.config.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        outputs = (hidden_states,)
        if self.config.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.config.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs


class CustomT5(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = CustomT5Stack(config, self.shared)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        emb_meta=None,
        **kwargs
    ):
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            emb_meta=emb_meta,
            **kwargs
        )

        return encoder_outputs
