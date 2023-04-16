from torch.optim import AdamW
from tqdm import tqdm
import torch


def train(
    model, dataloader_train, dataloader_validation, device, epochs=5, learning_rate=2e-5
):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for batch in tqdm(dataloader_train, desc=f"Epoch {epoch + 1}"):
            input_ids = batch["input_ids"].to(device)
            if "attention_mask" in batch.keys():  # Transformer models
                attention_mask = batch["attention_mask"].to(device)
            else:
                attention_mask = None
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = (
                model(input_ids)
                if attention_mask is None
                else model(input_ids, attention_mask)
            )
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(
            f"Training Loss after epoch {epoch + 1}: {epoch_loss / len(dataloader_train)}"
        )
        accuracy = evaluate(model, dataloader_validation, device)
        print("Validation accuracy: %s" % (accuracy,))


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            if "attention_mask" in batch.keys():  # Transformer models
                attention_mask = batch["attention_mask"].to(device)
            else:
                attention_mask = None
            labels = batch["labels"].to(device)
            logits = (
                model(input_ids)
                if attention_mask is None
                else model(input_ids, attention_mask)
            )
            predictions = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total
    return accuracy
