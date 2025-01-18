from typing import Any, List, Optional

import torch
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
        task_names_list: Optional[List[str]] = None,
        max_length: int = 128,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        self.max_length = max_length
        self.concat_metadata = concat_metadata
        self.metadatas = metadatas
        self.task_names_list = task_names_list
        if concat_metadata:
            self.x_data = [f"{x}. {m}" for x, m in zip(self.x_data, self.metadatas)]
        else:
            raise NotImplementedError(
                f"not implemented for non-concatened case in omnipred"
            )

    def __len__(self):
        return len(self.x_data)

    def _tokenize_and_pad(self, text: str) -> torch.Tensor:
        tokens = self.input_tokenizer.encode(text, add_bos=True, add_eos=True)
        pad_length = 0
        tokens_length = len(tokens)
        if tokens_length > self.max_length:
            tokens = tokens[: self.max_length]
            tokens_length = self.max_length

        elif tokens_length < self.max_length:
            pad_length = self.max_length - tokens_length
            tokens = tokens + [0] * pad_length

        return torch.tensor(tokens), pad_length, tokens_length

    def get_space_patch_start_idx(self, text: str, tokens_length: int) -> torch.Tensor:
        marker = torch.zeros(self.max_length, dtype=torch.int64)
        char_tensor = torch.tensor([ord(c) for c in text[:tokens_length]])
        space_idx = torch.where(char_tensor == 32)[0]
        marker[space_idx] = 1
        marker[tokens_length - 1] = 1
        marker[0] = 1
        marker = marker.cumsum(0)
        patch_num = marker.max() - 1
        return marker, patch_num

    def __getitem__(self, idx: int):
        x = str(self.x_data[idx])
        y = str(self.y_data[idx])

        # Encode input sequence
        x_tokens, pad_length, tokens_length = self._tokenize_and_pad(x)
        space_patch_start_idx, patch_num = self.get_space_patch_start_idx(
            x, tokens_length
        )

        y_tokens, _, _ = self._tokenize_and_pad(y)

        # Create decoder_input_ids
        decoder_input_ids = y_tokens["input_ids"].clone()
        decoder_input_ids = self._shift_right(
            decoder_input_ids.squeeze(),
            self.output_tokenizer.pad_token_id,
            self.output_tokenizer.decoder_start_token_id,
        )

        # For T5, we need to replace padding tokens in labels with -100
        labels = y_tokens["input_ids"].squeeze()
        labels[labels == self.output_tokenizer.pad_token_id] = 0

        return {
            "input_ids": x_tokens["input_ids"].squeeze(),
            "attention_mask": x_tokens["attention_mask"].squeeze(),
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": y_tokens["attention_mask"].squeeze(),
            "labels": labels,
            "task_name": self.task_names_list[idx],
        }

    def _shift_right(self, input_ids, pad_token_id, decoder_start_token_id):
        """Shift decoder input ids right by prepending decoder_start_token_id."""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[0] = decoder_start_token_id
        shifted_input_ids[1:] = input_ids[:-1].clone()

        # Replace possible -100 values in input_ids with pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
