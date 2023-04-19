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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs):
        # inputs is a list of tokenized utterances
        embeddings = self.embedding_layer(inputs)

        # Average the embeddings
        # Remove padding tokens (all-zero rows) and calculate the average for each utterance
        # TODO: Replace this by a mask
        averaged_embeddings = []
        for emb in embeddings:
            non_zero_emb = list(filter(lambda x: torch.any(x != 0), emb))
            if non_zero_emb:
                non_zero_emb = torch.stack(non_zero_emb)
                avg_emb = torch.mean(non_zero_emb, dim=0)
            else:
                avg_emb = torch.zeros(self.embedding_layer.embedding_dim).to(
                    self.device
                )
            averaged_embeddings.append(avg_emb)

        # Convert the list of averaged embeddings to a tensor
        averaged_embeddings = torch.stack(averaged_embeddings)

        # Pass the embeddings through the MLP layer
        final_out = self.network(averaged_embeddings)

        return final_out
