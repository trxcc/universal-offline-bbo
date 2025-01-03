from typing import Any, List, Optional

from torch.utils.data import Dataset


class OmnipredDataset(Dataset):
    def __init__(
        self,
        x_data: List[str],
        y_data: List[str],
        input_tokenizer: Any,
        output_tokenizer: Any,
        concat_metadata: bool = True,
        metadatas: Optional[List[str]] = None,
        max_length: int = 128,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.max_length = max_length
        self.concat_metadata = concat_metadata
        self.metadatas = metadatas

        if concat_metadata:
            self.x_data = [f"{x}. {m}" for x, m in zip(self.x_data, self.metadatas)]
        else:
            raise NotImplementedError(
                f"not implemented for non-concatened case in omnipred"
            )

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx: int):
        x = str(self.x_data[idx])
        y = str(self.y_data[idx])

        x_tokens = self.input_tokenizer(
            x,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        y_tokens = self.output_tokenizer(
            y,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": x_tokens["input_ids"].squeeze(),
            "attention_mask": x_tokens["attention_mask"].squeeze(),
            "labels": y_tokens["input_ids"].squeeze(),
        }
