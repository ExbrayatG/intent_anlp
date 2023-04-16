import torch.nn as nn
from transformers import AutoModel


class DistilBERTModel(nn.Module):
    def __init__(self, dropout_perc=0.1) -> None:
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(dropout_perc)
        # The classification layer is actually supposed to be added later in the main script and not at initilization
        # This is because output_dim must first be computed before instantiating the classification layer
        self.classifier = None
        self.output_dim = self.bert.config.hidden_size

    def forward(self, input_ids, attention_mask):
        if self.classifier is None:
            raise ValueError("The classifier layer has not been initialized yet")

        output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        pooled_output = output.last_hidden_state[:, 0, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
