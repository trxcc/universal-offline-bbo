from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class TextValueDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        values: List[float],
        metadatas: Optional[List[str]] = None,
        task_names: Optional[List[str]] = None,  # 添加task_names参数
    ) -> None:
        self.texts = texts
        self.values = torch.tensor(values, dtype=torch.float32)
        self.values = (self.values - self.values.mean(dim=0)) / self.values.std(dim=0)
        self.metadatas = metadatas
        self.task_names = task_names  # 存储每个样本对应的task_name

    def __len__(self):
        return len(self.texts)

    def __getitem__(
        self, idx: int
    ) -> Tuple[str, torch.Tensor, str, str]:  # 修改返回类型
        text = self.texts[idx]
        value = self.values[idx]
        metadata = self.metadatas[idx]
        task_name = self.task_names[idx]

        return text, value, metadata, task_name


if __name__ == "__main__":
    texts = ["hello", "world", "python"]
    values = [1.0, 2.5, 3.7]

    dataset = TextValueDataset(texts, values)
