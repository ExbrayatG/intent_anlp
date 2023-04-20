import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, embedding_layer, num_classes, num_filters=100, kernel_size=3):
        super(CNN, self).__init__()
        self.num_classes = num_classes

        self.embedding_layer = embedding_layer
        self.cnn_layer = nn.Conv1d(
            in_channels=self.embedding_layer.embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
        )
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc_layer = nn.Linear(num_filters, self.num_classes)

    def forward(self, inputs):
        # inputs is a list of tokenized utterances
        embeddings, mask = self.embedding_layer(inputs)

        # Apply the CNN layer to the embeddings
        cnn_out = self.cnn_layer(embeddings.transpose(1, 2))
        cnn_out = torch.relu(cnn_out)

        # Apply global max-pooling
        pooled_out = self.global_max_pool(cnn_out).squeeze(2)

        # Pass the pooled output through the fully connected layer
        final_out = self.fc_layer(pooled_out)

        return final_out
