import torch
import torch.nn as nn
from torchtext.vocab import FastText


class FastTextLayer(nn.Module):
    def __init__(self):
        super(FastTextLayer, self).__init__()
        self.fasttext_vectors = FastText(language="simple")
        self.embedding_dim = 300  # FastText's hidden size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs):
        # Get embeddings for each utterance in the batch
        embeddings = [
            self.fasttext_vectors.get_vecs_by_tokens(utterance.split())
            for utterance in inputs
        ]

        # Create a mask to track non-padding tokens
        mask = [torch.ones(len(utterance.split())) for utterance in inputs]
        mask = nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)

        # Pad the sequences to the same length
        embeddings = nn.utils.rnn.pad_sequence(embeddings, batch_first=True)

        return embeddings.to(self.device), mask.to(self.device)
