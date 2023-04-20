import torch.nn as nn
import torch


class ZeroLayerMLP(nn.Module):
    def __init__(self, embedding_layer, num_classes):
        super(ZeroLayerMLP, self).__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_layer = embedding_layer
        self.network = nn.Linear(self.embedding_layer.embedding_dim, self.num_classes)

    def forward(self, inputs):
        # inputs is a list of tokenized utterances
        embeddings, mask = self.embedding_layer(inputs)

        # Average the embeddings
        if mask is not None:
            # Calculate the sum of embeddings
            summed_embeddings = torch.sum(embeddings * mask.unsqueeze(-1), dim=1)

            # Calculate the number of tokens in each utterance
            token_count = torch.sum(mask, dim=1, keepdim=True)

            # Calculate the average of embeddings
            averaged_embeddings = summed_embeddings / token_count
        else:
            # Convert the list of averaged embeddings to a tensor
            averaged_embeddings = torch.mean(embeddings, dim=1)

        # Pass the embeddings through the MLP layer
        final_out = self.network(averaged_embeddings.to(self.device))

        return final_out
