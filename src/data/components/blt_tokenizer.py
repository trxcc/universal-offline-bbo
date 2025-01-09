# Copyright (c) Meta Platforms, Inc. and affiliates.
import abc


class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode(self, text: str, add_bos: bool, add_eos: bool):
        pass

    @abc.abstractmethod
    def decode(self, tokens):
        pass

    @abc.abstractmethod
    def get_token_offsets(self, text: str, tokens=None):
        """Return the offsets of the tokens in the original text. Only used for evaluation."""
        pass


class ByteTokenizer(Tokenizer):
    def __init__(self):
        self.bos_id = 256
        self.eos_id = 257
        self.n_words = 258

    def encode(self, s: str, add_bos: bool = False, add_eos: bool = False):
        tokens = [self.bos_id] * add_bos + list(s.encode()) + [self.eos_id] * add_eos
        return tokens

    def decode(self, tokens):
        byte_tokens = bytes([t for t in tokens if t < 256])
        return byte_tokens.decode("utf-8", errors="backslashreplace")

    def get_token_offsets(self, text: str, tokens=None):
        if tokens is None:
            tokens = self.encode(text)

        decoded_chars, offsets = [], []
        byte_pos = 0
        for token in tokens:
            if token < 256:
                char = bytes([token]).decode("utf-8", errors="ignore")
                if char:
                    decoded_chars.append(char)
                    offsets.append(byte_pos)
                byte_pos += len(char.encode("utf-8"))

        return decoded_chars, offsets


if __name__ == "__main__":
    tokenizer = ByteTokenizer()
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    print(tokens)
    print(tokenizer.decode(tokens))
    print(tokenizer.get_token_offsets(text))
