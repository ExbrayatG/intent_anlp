import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import torch


class DistilBertLayer(nn.Module):
    def __init__(self):
        super(DistilBertLayer, self).__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.embedding_dim = 768  # DistilBert's hidden size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs):
        # inputs is a list of utterances
        encoded_inputs = self.tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**encoded_inputs)[
            0
        ]  # Only retrieve the token embeddings, not the attention masks
        mask = None
        return outputs, mask
