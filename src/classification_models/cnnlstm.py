import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        embedding_layer,
        num_classes,
        hidden_dim=300,
        num_filters=100,
        kernel_size=3,
    ):
        super(CNNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.embedding_layer = embedding_layer
        self.cnn_layer = nn.Conv1d(
            in_channels=self.embedding_layer.embedding_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
        )
        self.lstm_layer = nn.LSTM(
            input_size=num_filters,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )
        self.fc_layer = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, inputs):
        # inputs is a list of tokenized utterances
        embeddings, mask = self.embedding_layer(inputs)

        # Apply the CNN layer to the embeddings
        cnn_out = self.cnn_layer(embeddings.transpose(1, 2))
        cnn_out = torch.relu(cnn_out)
        cnn_out = cnn_out.transpose(1, 2)

        # Pass the CNN output through the LSTM layer
        lstm_out, _ = self.lstm_layer(cnn_out)

        # Pass the final LSTM state through the fully connected layer
        final_out = self.fc_layer(lstm_out[:, -1, :])

        return final_out
