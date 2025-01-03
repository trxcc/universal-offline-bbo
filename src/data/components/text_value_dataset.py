from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class TextValueDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        values: List[float],
        metadatas: Optional[List[str]] = None,
    ) -> None:
        self.texts = texts
        self.values = torch.tensor(values, dtype=torch.float32)
        self.values = (self.values - self.values.mean(dim=0)) / self.values.std(dim=0)
        self.metadatas = metadatas

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        text, value, metadata = self.texts[idx], self.values[idx], self.metadatas[idx]

        return text, value, metadata


if __name__ == "__main__":
    texts = ["hello", "world", "python"]
    values = [1.0, 2.5, 3.7]

    dataset = TextValueDataset(texts, values)
