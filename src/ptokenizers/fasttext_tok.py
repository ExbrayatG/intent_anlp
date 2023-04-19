from .base_tokenizer import BaseTokenizer
from torchtext.vocab import FastText


class FastTextTokenizer(BaseTokenizer):
    """This class actually directly computes embeddings. This avoids creating and using a vocabulary, which was getting in the way of code modularity"""

    def __init__(self) -> None:
        super().__init__()
        self.fasttext_embedding = FastText("simple")

    def encode(self, dataset):
        def tokenize(sentence):
            return sentence.split()

        def get_avg_embedding(text):
            embeddings = [self.fasttext_embedding[word] for word in tokenize(text)]
            return sum(embeddings) / len(embeddings)

        return {"input_ids": [get_avg_embedding(text) for text in dataset]}
