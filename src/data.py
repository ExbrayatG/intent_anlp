import datasets
from torch.utils.data import DataLoader


def create_dataloaders(dataset, batch_size=16):
    train_dataloader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        dataset["validation"], batch_size=batch_size, shuffle=True
    )
    test_dataloader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def load_dataset(name: str):
    return datasets.load_dataset("silicone", name)
