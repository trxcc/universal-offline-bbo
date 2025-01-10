from typing import Any, Callable, List, Optional, Tuple

import torch

# from blt_tokenizer import ByteTokenizer
from torch.utils.data import DataLoader, Dataset


class EntropyDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        values: List[float],
        tokenizer: Any,
        tokenizer_max_length: int = 128,
        concat_metadata: bool = True,
        metadatas: Optional[List[str]] = None,
        task_names: Optional[List[str]] = None,
    ) -> None:
        self.texts = texts
        self.values = torch.tensor(values, dtype=torch.float32)
        self.values = (self.values - self.values.mean(dim=0)) / self.values.std(dim=0)
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.metadatas = metadatas
        self.task_names = task_names

        self.concat_metadata = concat_metadata
        if concat_metadata:
            self.texts = [f"{x}. {m}" for x, m in zip(self.texts, self.metadatas)]

    def __len__(self):
        return len(self.texts)

    def _tokenize_and_pad(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer.encode(text, add_bos=True, add_eos=True)

        if len(tokens) > self.tokenizer_max_length:
            tokens = tokens[: self.tokenizer_max_length]

        if len(tokens) < self.tokenizer_max_length:
            tokens = tokens + [0] * (self.tokenizer_max_length - len(tokens))

        return torch.tensor(tokens)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, str, str]:
        text = self.texts[idx]
        text_tokens = self._tokenize_and_pad(text)

        value = self.values[idx]

        metadata = self.metadatas[idx]
        metadata_tokens = self._tokenize_and_pad(metadata)

        task_names = self.task_names[idx]

        return {
            "text": text_tokens,
            "value": value.squeeze(),
            "metadata": metadata_tokens,
            "task_names": task_names,
        }


# if __name__ == "__main__":
#     texts = ["hello", "world", "python"]
#     values = [1.0, 2.5, 3.7]
#     metadatas = ["meta1", "meta2", "meta3"]
#     task_names = ["task1", "task2", "task3"]

#     dataset = EntropyDataset(
#         texts, values, ByteTokenizer(), metadatas=metadatas, task_names=task_names
#     )
#     dataloader = DataLoader(dataset, batch_size=2)
#     for batch in dataloader:
#         print(batch)
