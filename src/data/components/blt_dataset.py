from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from blt_tokenizer import ByteTokenizer

import torch.nn.functional as F

class BLTDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        values: List[float],
        tokenizer: Any,
        entropy_model: Any,
        entropy_threshold: float,
        tokenizer_max_length: int = 4096,
        concat_metadata: bool = True,
        metadatas: Optional[List[str]] = None,
        task_names: Optional[List[str]] = None,
    ) -> None:
        self.texts = texts
        self.values = torch.tensor(values, dtype=torch.float32)
        self.values = (self.values - self.values.mean(dim=0)) / self.values.std(dim=0)
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.entropy_model = entropy_model
        self.metadatas = metadatas
        self.task_names = task_names
        self.entropy_threshold = entropy_threshold

        self.concat_metadata = concat_metadata
        if concat_metadata:
            self.texts = [f"{x}. {m}" for x, m in zip(self.texts, self.metadatas)]

    def __len__(self):
        return len(self.texts)
    
    def _tokenize_and_pad(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        pad_length = 0
        tokens_length = len(tokens)
        if tokens_length > self.tokenizer_max_length:
            tokens = tokens[: self.tokenizer_max_length]
            tokens_length = self.tokenizer_max_length

        if tokens_length < self.tokenizer_max_length:
            pad_length = self.tokenizer_max_length - tokens_length
            tokens = tokens + [0] * pad_length

        return torch.tensor(tokens), pad_length, tokens_length

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, str, str]:
        text = self.texts[idx]
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        for k, v in text_tokens.items():
            text_tokens[k] = v.squeeze()

        value = self.values[idx]

        metadata = self.metadatas[idx]
        metadata_tokens = self.tokenizer(
            metadata,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )
        for k, v in metadata_tokens.items():
            metadata_tokens[k] = v.squeeze()

        task_names = self.task_names[idx]
        entropy_patch_start_idx = self.get_entropy_patch_start_idx(idx)

        return {
            "text": text_tokens,
            "value": value.squeeze(),
            "metadata": metadata_tokens,
            "task_names": task_names,
            "entropy_patch_start_idx": entropy_patch_start_idx,
        }
    
    def get_entropy_patch_start_idx(self, idx: int) -> torch.Tensor:
        text = self.texts[idx]
        text_tokens, pad_length, tokens_length = self._tokenize_and_pad(text)
        logits = self.entropy_model(text_tokens)
        logits = logits.reshape(-1, logits.shape[-1])[
                : tokens_length - pad_length, :
            ]
        entropy = self.entropy(logits)
        start_idx = self.get_entropy_patch_start_idx(entropy, self.entropy_threshold)
        return start_idx

    def get_entropy_patch_start_idx(self, entropy: torch.Tensor, threshold: float) -> torch.Tensor:
        start_idx = torch.zeros_like(entropy, dtype=torch.bool)
        start_idx[:, 0] = True
        diff = entropy[:, 1:] - entropy[:, :-1]
        start_idx[:, 1:] = diff > threshold
        return start_idx


    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
        return entropy

if __name__ == "__main__":
    texts = ["hello", "world", "python"]
    values = [1.0, 2.5, 3.7]
    metadatas = ["meta1", "meta2", "meta3"]
    task_names = ["task1", "task2", "task3"]

    dataset = BLTDataset(
        texts, values, ByteTokenizer(), metadatas=metadatas, task_names=task_names
    )
    dataloader = DataLoader(dataset, batch_size=2)
    for batch in dataloader:
        print(batch)