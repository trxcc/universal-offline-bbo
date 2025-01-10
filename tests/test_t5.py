# from transformers import T5ForConditionalGeneration, T5Tokenizer

# tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-small")
# model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

# task_prefix = "translate English to German: "
# # use different length sentences to test batching
# sentences = ["The house is wonderful.", "I like to work in NYC."]

# inputs = tokenizer(
#     [task_prefix + sentence for sentence in sentences],
#     return_tensors="pt",
#     padding=True,
# )
# print(inputs)

# output_sequences = model.generate(
#     input_ids=inputs["input_ids"],
#     attention_mask=inputs["attention_mask"],
#     do_sample=False,  # disable sampling to test if batching affects output
# )

# print(tokenizer.batch_decode(output_sequences, skip_special_tokens=True))

from transformers import T5EncoderModel, T5Model, T5Tokenizer

# model = T5Model.from_pretrained("t5-small")
model = T5EncoderModel.from_pretrained("t5-small")
tok = T5Tokenizer.from_pretrained("t5-small")

enc = tok("some text", return_tensors="pt")
print(enc["input_ids"].shape)
emb = model.encoder(
    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True
).last_hidden_state
print(emb.shape)
assert 0

# forward pass through encoder only
output = model.encoder(
    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], return_dict=True
)
# get the final hidden states
emb = output.last_hidden_state
print(emb)
