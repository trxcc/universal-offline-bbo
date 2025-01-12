from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.components.blt_tokenizer import ByteTokenizer
from src.models.components.entropy_model import ByteTransformer


class BLTDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        values: List[float],
        tokenizer: Any,
        entropy_model: Any,
        entropy_model_checkpoint: str,
        entropy_threshold: float,
        tokenizer_max_length: int = 2048,
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
        self.entropy_model.load_from_checkpoint(entropy_model_checkpoint)
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
            tokens_length = pad_length

        elif tokens_length < self.tokenizer_max_length:
            pad_length = self.tokenizer_max_length - tokens_length
            tokens = tokens + [0] * pad_length

        return torch.tensor(tokens), pad_length, tokens_length

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, str, str]:
        text = self.texts[idx]
        text_tokens, _, _ = self._tokenize_and_pad(text)

        value = self.values[idx]

        metadata = self.metadatas[idx]
        metadata_tokens, _, _ = self._tokenize_and_pad(metadata)

        task_names = self.task_names[idx]
        entropy_patch_start_idx = self.get_entropy_patch_start_idx(text)

        return {
            "text": text_tokens,
            "value": value.squeeze(),
            "metadata": metadata_tokens,
            "task_names": task_names,
            "entropy_patch_start_idx": entropy_patch_start_idx,
        }

    def get_entropy_patch_start_idx(self, text: str) -> torch.Tensor:
        text_tokens, pad_length, tokens_length = self._tokenize_and_pad(text)
        logits = self.entropy_model.get_single_logits(text_tokens)
        logits = logits.reshape(-1, logits.shape[-1])
        entropy = self.entropy(logits)
        start_idx = self.get_entropy_patch_idx(entropy, self.entropy_threshold, tokens_length)
        return start_idx

    def get_entropy_patch_idx(
        self, entropy: torch.Tensor, threshold: float, tokens_length: torch.Tensor
    ) -> torch.Tensor:
    # entropy: [bsz, seq_len]
    # tokens_length: [bsz, 1]
        entropy = entropy.unsqueeze(0)
        tokens_length = torch.tensor(tokens_length)
        tokens_length = tokens_length.unsqueeze(0)
        bsz, seq_len = entropy.shape
    
    # 初始化start_idx
        start_idx = torch.zeros_like(entropy, dtype=torch.bool)  # [bsz, seq_len]
        start_idx[:, 0] = True

    # 计算差值并标记阈值点
        diff = entropy[:, 1:] - entropy[:, :-1]
        start_idx[:, 1:] = diff > threshold
    
    # 在tokens_length处标记结束
        batch_indices = torch.arange(bsz, device=entropy.device)
        start_idx[batch_indices, tokens_length.squeeze(-1)-1] = True
    
    # 将bool tensor转为int64
        start_idx_int = start_idx.long()  # [bsz, seq_len]
    
    # 计算累积和来获得patch索引
        result = start_idx_int.cumsum(dim=1) - 1  # [bsz, seq_len]
        print(result)
        print(result.shape)
        assert False
    
        return result

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
        return entropy


if __name__ == "__main__":
    texts = ["hello", "world", "python"]
    values = [1.0, 2.5, 3.7]
    metadatas = ["meta1", "meta2", "meta3"]
    task_names = ["task1", "task2", "task3"]
    entropy_threshold = 0.5
    tokenizer_max_length = 4096
    tokenizer = ByteTokenizer()
    entropy_model = ByteTransformer()
    entropy_model.load_from_checkpoint(
        "/root/autodl-tmp/universal_offline/universal-offline-bbo/logs/entropy_model/runs/2025-01-10_23-44-21_seed42/checkpoints/last.ckpt"
    )

    dataset = BLTDataset(
        texts,
        values,
        tokenizer=tokenizer,
        entropy_model=entropy_model,
        entropy_model_checkpoint="/root/autodl-tmp/universal_offline/universal-offline-bbo/logs/entropy_model/runs/2025-01-10_23-44-21_seed42/checkpoints/last.ckpt",
        entropy_threshold=entropy_threshold,
        metadatas=metadatas,
        task_names=task_names,
    )
    dataloader = DataLoader(dataset, batch_size=3)
    for batch in dataloader:
        print(batch)
