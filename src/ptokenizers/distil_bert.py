from .base_tokenizer import BaseTokenizer
from transformers import AutoTokenizer


class DistilBERTTokenizer:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def encode(self, dataset):
        return self.tokenizer(dataset, truncation=True, padding=True)
