import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self, embedding_layer, num_classes, hidden_dim=300):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.embedding_layer = embedding_layer
        self.lstm_layer = nn.LSTM(
            input_size=self.embedding_layer.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.fc_layer = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, inputs):
        # inputs is a list of tokenized utterances
        embeddings, mask = self.embedding_layer(inputs)

        # Pass the embeddings through the bidirectional LSTM layer
        lstm_out, _ = self.lstm_layer(embeddings)

        # Concatenate the final states from both directions and pass it through the fully connected layer
        final_out = self.fc_layer(lstm_out[:, -1, :])

        return final_out
