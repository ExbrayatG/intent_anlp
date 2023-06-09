from torch.optim import AdamW
from tqdm import tqdm
import torch
import wandb


def train(
    model,
    dataloader_train,
    dataloader_validation,
    device,
    epochs=5,
    learning_rate=2e-5,
    log_wandb=False,
):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for batch in tqdm(dataloader_train, desc=f"Epoch {epoch + 1}"):
            utterances = batch["Utterance"]

            labels = batch["Label"].to(device)

            optimizer.zero_grad()
            logits = model(utterances)
            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        training_loss = epoch_loss / len(dataloader_train)
        print(f"Training Loss after epoch {epoch + 1}: {training_loss:.4f}")
        accuracy, validation_loss = evaluate(model, dataloader_validation, device)
        print("Validation accuracy: %s" % (accuracy,))
        if log_wandb:
            wandb.log(
                {
                    "training_loss": training_loss,
                    "validation_accuracy": accuracy,
                    "validation_loss": validation_loss,
                }
            )


def evaluate(model, dataloader, device):
    model.to(device)
    model.eval()
    total = 0
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            utterances = batch["Utterance"]
            labels = batch["Label"].to(device)
            logits = model(utterances)
            predictions = torch.argmax(logits, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            loss = torch.nn.CrossEntropyLoss()(logits, labels)
            total_loss += loss.item()

    accuracy = correct / total
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss
