"""Dataset classes for fine-tuning."""

import torch
from torch.utils.data import Dataset


class TextChunkDataset(Dataset):
    """Tokenizes a text into overlapping chunks for causal LM training."""

    def __init__(self, text: str, tokenizer, max_length: int = 512, stride: int = 256):
        self.examples = []
        orig_max = tokenizer.model_max_length
        tokenizer.model_max_length = int(1e9)
        tokens = tokenizer.encode(text)
        tokenizer.model_max_length = orig_max
        for i in range(0, max(1, len(tokens) - max_length + 1), stride):
            chunk = tokens[i : i + max_length]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))
        # Include final chunk if it wasn't covered
        if len(tokens) > max_length and (len(tokens) - max_length) % stride != 0:
            chunk = tokens[-max_length:]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        input_ids = self.examples[idx]
        return {"input_ids": input_ids, "labels": input_ids.clone()}
