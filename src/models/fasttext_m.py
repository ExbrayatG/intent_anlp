import torch.nn as nn
import torch


class FastTextModel(nn.Module):
    # As the tokenizer for fasttext actually computes the average embedding, this class is just the identity
    def __init__(self) -> None:
        super().__init__()
        self.classifier = None
        self.output_dim = 300  # This is hardcoded because fasttext pretrained embeddings always have a dimension of 300 (we won't try to redimension it in this project)

    def forward(self, x):
        return self.classifier(x)
