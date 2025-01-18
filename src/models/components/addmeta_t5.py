import torch
import torch.nn as nn
from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Block, T5Stack


class CustomT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

    def forward(self, input_ids=None, attention_mask=None, emb_meta=None, **kwargs):

        meta_mask = attention_mask.unsqueeze(2)

        if attention_mask is not None:
            batch_size = attention_mask.size(0)
            # 获取注意力头数
            num_heads = self.config.num_heads
            # 扩展attention_mask为4D tensor: [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.expand(
                batch_size, num_heads, attention_mask.size(1), attention_mask.size(1)
            )
            # 将0和1转换为大的负值和0
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_mask = extended_attention_mask

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        all_hidden_states = () if self.config.output_hidden_states else None
        all_attentions = () if self.config.output_attentions else None

        # 处理emb_meta
        processed_emb_meta = None
        if emb_meta is not None:
            batch_size, seq_length, hidden_dim = hidden_states.size()
            if len(emb_meta.size()) == 2:
                processed_emb_meta = emb_meta.unsqueeze(1).expand(
                    batch_size, seq_length, hidden_dim
                )
            else:
                processed_emb_meta = emb_meta

            if attention_mask is not None:
                attention_mask_float = meta_mask.float()
                processed_emb_meta = processed_emb_meta * attention_mask_float

        for i, block in enumerate(self.block):
            layer_outputs = block(
                hidden_states, attention_mask=attention_mask, **kwargs
            )
            hidden_states = layer_outputs[0]

            if processed_emb_meta is not None:
                hidden_states = hidden_states + processed_emb_meta

            if self.config.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.config.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1:])

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
