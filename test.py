import torch
import torch.nn as nn
from transformers import T5Config, T5PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Block, T5Stack


class CustomT5Stack(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        emb_meta=None,  # 添加metadata embedding参数
        **kwargs
    ):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)

        all_hidden_states = () if self.config.output_hidden_states else None
        all_attentions = () if self.config.output_attentions else None

        # 确保emb_meta的维度与hidden_states匹配
        # 假设emb_meta shape为 [batch_size, meta_dim]
        # 需要扩展到 [batch_size, seq_len, meta_dim]
        if emb_meta is not None:
            emb_meta = emb_meta.unsqueeze(1).expand(-1, hidden_states.size(1), -1)

        for i, block in enumerate(self.block):
            hidden_states = block(
                hidden_states, attention_mask=attention_mask, **kwargs
            )[0]

            # 在每一层的输出上添加metadata信息
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
        self.decoder = T5Stack(config, self.shared)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        emb_meta=None,  # metadata embedding
        **kwargs
    ):
        # 编码器前向传播，包含metadata
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            emb_meta=emb_meta,
            **kwargs
        )

        # 解码器前向传播
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            **kwargs
        )

        return decoder_outputs


# 初始化模型
config = T5Config.from_pretrained("t5-base")
print(config)
model = CustomT5(config)

# 准备输入
input_ids = torch.LongTensor([[1, 2, 3, 4]])
attention_mask = torch.ones_like(input_ids)
emb_meta = torch.randn(1, config.d_model)  # [batch_size, d_model]

# 前向传播
encoder_outputs = model.encoder(
    input_ids=input_ids, attention_mask=attention_mask, emb_meta=emb_meta
)

print(encoder_outputs)
