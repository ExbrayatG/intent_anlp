import torch.nn as nn
import torch


class OneLayerMLP(nn.Module):
    def __init__(self, embedding_layer, num_classes, hidden_dim=300):
        super(OneLayerMLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.embedding_layer = embedding_layer

        self.network = nn.Sequential(
            nn.Linear(self.embedding_layer.embedding_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    def forward(self, inputs):
        # inputs is a list of tokenized utterances
        embeddings = self.embedding_layer(inputs)

        # Average the embeddings
        average_embeddings = torch.mean(embeddings, dim=1)  # TODO - check this

        # Pass the embeddings through the MLP layer
        final_out = self.network(average_embeddings)

        return final_out
