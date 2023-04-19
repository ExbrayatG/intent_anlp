import torch
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch.nn as nn
from transformers import AutoModel
from transformers import AdamW
from tqdm import tqdm


class DialogueDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class BaseClassifier(nn.Module):
    def __init__(self, pretrained_model_name, num_labels):
        super(BaseClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )

        if hasattr(output, "pooler_output"):
            pooled_output = output.pooler_output
        else:
            # Use the last hidden state (usually for DistilBERT)
            last_hidden_state = output.last_hidden_state
            pooled_output = last_hidden_state[:, 0, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def load_datasets():
    silicone_swda = load_dataset("silicone", "swda")
    silicone_mrda = load_dataset("silicone", "mrda")
    return silicone_swda, silicone_mrda


def print_dataset_metrics(silicone_mrda):
    print("MRDA train labels: ", Counter(silicone_mrda["train"]["Label"]))
    print("MRDA test labels: ", Counter(silicone_mrda["test"]["Label"]))
    print("MRDA sample: ", silicone_mrda["train"][5])
    print(
        "MRDA max utterance length",
        max([len(ut) for ut in silicone_mrda["train"]["Utterance"]]),
    )
    print(
        "MRDA average utterance length: ",
        sum([len(ut) for ut in silicone_mrda["train"]["Utterance"]])
        / len([len(ut) for ut in silicone_mrda["train"]["Utterance"]]),
    )


def preprocess_and_tokenize(silicone_swda, silicone_mrda):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    swda_encodings = tokenizer(
        silicone_swda["train"]["Utterance"], truncation=True, padding=True
    )
    mrda_encodings = tokenizer(
        silicone_mrda["train"]["Utterance"], truncation=True, padding=True
    )

    return swda_encodings, mrda_encodings


def create_dataloaders(
    swda_encodings, mrda_encodings, silicone_swda, silicone_mrda, batch_size=16
):
    swda_dataset = DialogueDataset(swda_encodings, silicone_swda["train"]["Label"])
    mrda_dataset = DialogueDataset(mrda_encodings, silicone_mrda["train"]["Label"])

    swda_dataloader = DataLoader(swda_dataset, batch_size=batch_size, shuffle=True)
    mrda_dataloader = DataLoader(mrda_dataset, batch_size=batch_size, shuffle=True)

    return swda_dataloader, mrda_dataloader


def train(model, dataloader, device, epochs=5, learning_rate=2e-5):
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Loss after epoch {epoch + 1}: {epoch_loss / len(dataloader)}")


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total
    print(f"Accuracy: {accuracy}")


if __name__ == "__main__":
    silicone_swda, silicone_mrda = load_datasets()
    print_dataset_metrics(silicone_mrda)
    swda_encodings, mrda_encodings = preprocess_and_tokenize(
        silicone_swda, silicone_mrda
    )
    swda_dataloader, mrda_dataloader = create_dataloaders(
        swda_encodings, mrda_encodings, silicone_swda, silicone_mrda
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_model_name = "distilbert-base-uncased"

    # Train and evaluate on SWDA
    swda_model = BaseClassifier(
        pretrained_model_name,
        num_labels=len(silicone_swda["train"].features["Label"].names),
    )
    train(swda_model, swda_dataloader, device)
    evaluate(swda_model, swda_dataloader, device)

    # Train and evaluate on MRDA
    mrda_model = BaseClassifier(
        pretrained_model_name,
        num_labels=len(silicone_mrda["train"].features["Label"].names),
    )
    train(mrda_model, mrda_dataloader, device)
    evaluate(mrda_model, mrda_dataloader, device)
