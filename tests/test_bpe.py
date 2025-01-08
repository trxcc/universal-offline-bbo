from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# tokenizer = Tokenizer.from_file("path/to/tokenizer.json")

# tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
# tokenizer.pre_tokenizer = Whitespace()

# trainer = BpeTrainer(
#     vocab_size=30000,
#     min_frequency=2,
#     special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
# )

# files = ["path/to/file1.txt", "path/to/file2.txt"]
# tokenizer.train(files, trainer)

# tokenizer.save("path/to/save/tokenizer.json")

text = "Hello, this is a test sentence!"
output = tokenizer.encode(text)
print(output)
