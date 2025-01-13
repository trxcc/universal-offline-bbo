from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# from src.data.components.blt_tokenizer import ByteTokenizer
# from src.models.components.entropy_model import ByteTransformer


class BLTSpaceDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        values: List[float],
        tokenizer: Any,
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
        # self.entropy_model = entropy_model
        # self.entropy_model.load_from_checkpoint(entropy_model_checkpoint)
        self.metadatas = metadatas
        self.task_names = task_names
        # self.entropy_threshold = entropy_threshold

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

        elif tokens_length < self.tokenizer_max_length:
            pad_length = self.tokenizer_max_length - tokens_length
            tokens = tokens + [0] * pad_length

        return torch.tensor(tokens), pad_length, tokens_length

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, str, str]:
        text = self.texts[idx]
        assert len(text) > 0
        text_tokens, pad_length, tokens_length = self._tokenize_and_pad(text)

        value = self.values[idx]

        metadata = self.metadatas[idx]
        metadata_tokens, _, _ = self._tokenize_and_pad(metadata)

        task_names = self.task_names[idx]
        space_patch_start_idx = self.get_space_patch_start_idx(text, tokens_length)

        return {
            "text": text_tokens,
            "value": value.squeeze(),
            "metadata": metadata_tokens,
            "task_names": task_names,
            "space_patch_start_idx": space_patch_start_idx,
        }

    def get_space_patch_start_idx(self, text: str, tokens_length: int) -> torch.Tensor:
        marker = torch.zeros(self.tokenizer_max_length, dtype=torch.int64)
        char_tensor = torch.tensor([ord(c) for c in text[:tokens_length]])
        space_idx = torch.where(char_tensor == 32)[0]
        marker[space_idx] = 1
        marker[tokens_length - 1] = 1
        marker[0] = 1
        marker = marker.cumsum(0)
        return marker


# if __name__ == "__main__":
#     texts = ["hello", "world", "python"]
#     values = [1.0, 2.5, 3.7]
#     metadatas = ["meta1", "meta2", "meta3"]
#     task_names = ["task1", "task2", "task3"]
#     entropy_threshold = 0.5
#     tokenizer_max_length = 4096
#     tokenizer = ByteTokenizer()
#     entropy_model = ByteTransformer()
#     entropy_model.load_from_checkpoint(
#         "/root/autodl-tmp/universal_offline/universal-offline-bbo/logs/entropy_model/runs/2025-01-10_23-44-21_seed42/checkpoints/last.ckpt"
#     )

#     dataset = BLTDataset(
#         texts,
#         values,
#         tokenizer=tokenizer,
#         entropy_model=entropy_model,
#         entropy_threshold=entropy_threshold,
#         metadatas=metadatas,
#         task_names=task_names,
#     )
#     dataloader = DataLoader(dataset, batch_size=2)
#     for batch in dataloader:
#         print(batch)
