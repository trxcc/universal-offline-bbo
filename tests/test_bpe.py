from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer

# 方法1: 使用预训练的 transformers tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")  # 或其他使用BPE的模型

# 方法2: 如果有已保存的自定义tokenizer文件
# tokenizer = Tokenizer.from_file("path/to/tokenizer.json")

# 方法3: 从头训练新的tokenizer
# tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
# tokenizer.pre_tokenizer = Whitespace()

# trainer = BpeTrainer(
#     vocab_size=30000,
#     min_frequency=2,
#     special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
# )

# 训练tokenizer (如果需要)
# files = ["path/to/file1.txt", "path/to/file2.txt"]
# tokenizer.train(files, trainer)

# 保存训练好的tokenizer
# tokenizer.save("path/to/save/tokenizer.json")

# 测试tokenizer
text = "Hello, this is a test sentence!"
output = tokenizer.encode(text)
print(output)
