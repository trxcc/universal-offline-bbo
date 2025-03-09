import torch
import torch.nn as nn
from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Block, T5Stack


class CustomT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)
        self.meta_len = 64
        self.seq_len = 324
        self.seq_len_reducers = nn.ModuleList([
            nn.Linear(self.meta_len+self.seq_len, self.seq_len) 
            for _ in range(len(self.block))
        ])

    def forward(self, input_ids=None, attention_mask=None, emb_meta=None, **kwargs):
        device = input_ids.device
        
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        if emb_meta is not None:
            emb_meta = emb_meta.to(device)

        if attention_mask is not None:
            batch_size = attention_mask.size(0)
            # 获取注意力头数
            num_heads = self.config.num_heads
            # 扩展attention_mask为4D tensor: [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask2 = attention_mask.unsqueeze(1).unsqueeze(-1)
            attention_scores_mask = attention_mask2 * extended_attention_mask
            extended_attention_mask = attention_scores_mask.expand(
                batch_size, num_heads, attention_mask.size(1), attention_mask.size(1)
            )
            # 将0和1转换为大的负值和0
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attention_mask = extended_attention_mask

        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        all_hidden_states = () if self.config.output_hidden_states else None
        all_attentions = () if self.config.output_attentions else None
        
        if attention_mask is not None:
            meta_seq_len = self.meta_len
            batch_size, num_heads, _, seq_len = attention_mask.size()
            new_attention_mask = torch.full(
                (batch_size, num_heads, seq_len + meta_seq_len, seq_len + meta_seq_len),
                0,
                device=attention_mask.device,
                dtype=attention_mask.dtype
            )
            new_attention_mask[:, :, meta_seq_len:, meta_seq_len:] = attention_mask
        for i, (block, reducer) in enumerate(zip(self.block, self.seq_len_reducers)):
            # print(type(emb_meta))
            # print(emb_meta.size())
            # print(hidden_states.size())
            # assert 0
            hidden_states = torch.cat([emb_meta, hidden_states], dim=1)
            layer_outputs = block(
                hidden_states, attention_mask=new_attention_mask, **kwargs
            )
            hidden_states = layer_outputs[0]

            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = reducer(hidden_states)
            hidden_states = hidden_states.transpose(1, 2)


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
# logs/embed_regress_t5_catmeta/runs/2025-01-25_18-44-14_seed42/checkpoints/last.ckpt