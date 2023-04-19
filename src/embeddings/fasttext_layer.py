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
        # inputs is a list of utterances
        inputs = [
            [x.lower() for x in elt.split()] for elt in inputs
        ]  # Simple tokenization
        batch_size = len(inputs)
        seq_lengths = [len(utterance) for utterance in inputs]

        # Get the FastText embeddings for each token
        embeddings = []
        for utterance in inputs:
            token_embeddings = []
            for token in utterance:
                token_embedding = self.fasttext_vectors[token]
                token_embeddings.append(token_embedding)
            embeddings.append(torch.stack(token_embeddings))

        # Pad the embeddings to make them the same length
        max_length = max(seq_lengths)
        padded_embeddings = []
        for i in range(batch_size):
            padding_length = max_length - seq_lengths[i]
            padding = torch.zeros(padding_length, self.embedding_dim)
            padded_embeddings.append(torch.cat([embeddings[i], padding], dim=0))

        # Stack the embeddings into a tensor
        stacked_embeddings = torch.stack(padded_embeddings).to(self.device)

        return stacked_embeddings
