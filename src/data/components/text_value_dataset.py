from typing import Any, Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.data.data_utils import normalize_ys_from_different_tasks


class TextValueDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        values: List[float],
        tokenizer: Any,
        tokenizer_max_length: int = 128,
        meta_tok_max_len: int = 64,
        concat_metadata: bool = True,
        cat_front: bool = True,
        metadatas: Optional[List[str]] = None,
        task_names: Optional[List[str]] = None,
    ) -> None:
        self.texts = texts
        # self.values = torch.tensor(values, dtype=torch.float32)
        # self.values = (self.values - self.values.mean(dim=0)) / self.values.std(dim=0)
        self.values = normalize_ys_from_different_tasks(values, task_names)
        self.tokenizer = tokenizer
        self.tokenizer_max_length = tokenizer_max_length
        self.metadatas = metadatas
        self.task_names = task_names
        self.meta_max_len = meta_tok_max_len

        self.concat_metadata = concat_metadata
        if concat_metadata:
            if cat_front:
                self.texts = [f"{m}. {x}" for x, m in zip(self.texts, self.metadatas)]
            else:
                self.texts = [f"{x}. {m}" for x, m in zip(self.texts, self.metadatas)]

    def __len__(self):
        return len(self.texts)
    
    def transform_text(self, text):
    # 找到第一个 x1 的位置作为分隔点
        split_point = text.find('x1:')
    
    # 分割metadata和数据部分
        metadata = text[:split_point].strip()
        data = text[split_point:].strip()
    
    # 处理数据部分
        items = data.split(',')  # 假设用逗号分隔
        transformed_items = []
    
        for item in items:
        # 分离特征名和数值
            feature, value = item.strip().split(':')
        # 转换格式
            transformed = f"{feature} is{value}"
            transformed_items.append(transformed)
    
    # 组合结果
        result = metadata + '\n' + '\n'.join(transformed_items)
        return result

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, str, str]:
        text = self.texts[idx]
        # text = self.transform_text(text)
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
            max_length=self.meta_max_len,
            truncation=True,
            return_tensors="pt",
        )
        for k, v in metadata_tokens.items():
            metadata_tokens[k] = v.squeeze()

        task_names = self.task_names[idx]

        return {
            "text": text_tokens,
            "value": value.squeeze(),
            "metadata": metadata_tokens,
            "task_names": task_names,
        }


if __name__ == "__main__":
    texts = ["hello", "world", "python"]
    values = [1.0, 2.5, 3.7]

    dataset = TextValueDataset(texts, values)
