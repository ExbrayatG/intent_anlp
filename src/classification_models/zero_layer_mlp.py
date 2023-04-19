import torch.nn as nn
import torch


class ZeroLayerMLP(nn.Module):
    def __init__(self, embedding_layer, num_classes):
        super(ZeroLayerMLP, self).__init__()
        self.num_classes = num_classes

        self.embedding_layer = embedding_layer
        self.network = nn.Linear(self.embedding_layer.embedding_dim, self.num_classes)

    def forward(self, inputs):
        # inputs is a list of tokenized utterances
        embeddings = self.embedding_layer(inputs)

        # Average the embeddings
        average_embeddings = torch.mean(embeddings, dim=1)  # TODO - check this

        # Pass the embeddings through the MLP layer
        final_out = self.network(average_embeddings)

        return final_out
