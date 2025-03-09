from typing import Any, List, Optional

import torch 
from torch.utils.data import Dataset

from src.data.data_utils import normalize_ys_from_different_tasks

class OmnipredDataset(Dataset):
    def __init__(
        self,
        x_data: List[str],
        y_data: List[str],
        poorest_data_size: int,
        input_tokenizer: Any,
        output_tokenizer: Any,
        concat_metadata: bool = True,
        cat_front: bool = True,
        metadatas: Optional[List[str]] = None,
        task_names_list: Optional[List[str]] = None,
        max_length: int = 128,
    ) -> None:
        # 将 y_data 转换为浮点数
        y_values = [float(y) for y in y_data]
        
        # 获取排序后的索引
        sorted_indices = sorted(range(len(y_values)), key=lambda k: y_values[k])
        
        # 选择最小的 poorest_data_size 个数据的索引
        selected_indices = sorted_indices[:poorest_data_size]
        
        # 根据选择的索引更新所有数据
        self.x_data = [x_data[i] for i in selected_indices]
        self.y_data = [y_data[i] for i in selected_indices]
        
        if metadatas is not None:
            self.metadatas = [metadatas[i] for i in selected_indices]
        else:
            self.metadatas = metadatas
            
        if task_names_list is not None:
            self.task_names_list = [task_names_list[i] for i in selected_indices]
        else:
            self.task_names_list = task_names_list
        
        self.values = normalize_ys_from_different_tasks(self.y_data, self.task_names_list)
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.max_length = max_length
        self.concat_metadata = concat_metadata
        
        if concat_metadata:
            if cat_front:
                self.x_data = [f"{m}. {x}" for x, m in zip(self.x_data, self.metadatas)]
            else:
                self.x_data = [f"{x}. {m}" for x, m in zip(self.x_data, self.metadatas)]
        else:
            pass

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx: int):
        x = str(self.x_data[idx])
        y = str(self.y_data[idx])
        value = self.values[idx]

        # Encode input sequence
        x_tokens = self.input_tokenizer(
            x,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Encode target sequence
        y_tokens = self.output_tokenizer(
            y,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        metadata = self.metadatas[idx]
        metadata_tokens = self.input_tokenizer(
            metadata,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        for k, v in metadata_tokens.items():
            metadata_tokens[k] = v.squeeze()

        # Create decoder_input_ids
        decoder_input_ids = y_tokens["input_ids"].clone()
        decoder_input_ids = self._shift_right(
            decoder_input_ids.squeeze(),
            self.output_tokenizer.pad_token_id,
            self.output_tokenizer.decoder_start_token_id,
        )

        # For T5, we need to replace padding tokens in labels with -100
        labels = y_tokens["input_ids"].squeeze()
        labels[labels == self.output_tokenizer.pad_token_id] = -100

        return {
            "input_ids": x_tokens["input_ids"].squeeze(),
            "attention_mask": x_tokens["attention_mask"].squeeze(),
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": y_tokens["attention_mask"].squeeze(),
            "labels": labels,
            "task_name": self.task_names_list[idx],
            "metadata": metadata_tokens,
            "value": value.squeeze(),
        }

    def _shift_right(self, input_ids, pad_token_id, decoder_start_token_id):
        """Shift decoder input ids right by prepending decoder_start_token_id."""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[0] = decoder_start_token_id
        shifted_input_ids[1:] = input_ids[:-1].clone()

        # Replace possible -100 values in input_ids with pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
