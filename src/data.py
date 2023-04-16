import datasets
from torch.utils.data import Dataset, DataLoader
import torch


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


def create_dataloaders(
    train_encodings, val_encodings, test_encodings, dataset, batch_size=16
):
    train_dataset = DialogueDataset(train_encodings, dataset["train"]["Label"])
    val_dataset = DialogueDataset(val_encodings, dataset["validation"]["Label"])
    test_dataset = DialogueDataset(test_encodings, dataset["test"]["Label"])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def load_dataset(name: str):
    return datasets.load_dataset("silicone", name)
