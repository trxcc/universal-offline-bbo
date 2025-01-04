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
        }

    def _shift_right(self, input_ids, pad_token_id, decoder_start_token_id):
        """Shift decoder input ids right by prepending decoder_start_token_id."""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[0] = decoder_start_token_id
        shifted_input_ids[1:] = input_ids[:-1].clone()

        # Replace possible -100 values in input_ids with pad_token_id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids
