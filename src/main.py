from data import load_dataset, create_dataloaders
import argparse
import torch
from embeddings import get_embedding_class, models
from classification_models import get_classification_model_class, layers
from utils import train, evaluate
from config import CLASSIFICATION_MODELS, EMBEDDINGS
import wandb


def main(args):
    wandb_config = {
        "embedding": args.embedding,
        "classification_model": args.classification_model,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "classification_model_config": CLASSIFICATION_MODELS.get(
            args.classification_model, {}
        ),
        "embedding_config": EMBEDDINGS.get(args.embedding, {}),
    }

    optional_args = {}
    if args.run_name is not None:
        optional_args["name"] = args.run_name
    if args.run_group is not None:
        optional_args["group"] = args.run_group

    if args.wandb:
        wandb.init(
            project="intent-anlp",
            entity="projet-ovations",
            config=wandb_config,
            **optional_args
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.verbose:
        print("Running on device: ", device)

    # Load the data
    if args.verbose:
        print("Loading dataset")
    dataset = load_dataset(args.dataset)

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        dataset, args.batch_size
    )

    embedding_layer = get_embedding_class(args.embedding)(
        **EMBEDDINGS.get(args.embedding, {})
    )

    num_labels = len(dataset["train"].features["Label"].names)

    classification_model = get_classification_model_class(args.classification_model)(
        embedding_layer,
        num_labels,
        **CLASSIFICATION_MODELS.get(args.classification_model, {})
    )

    # Train the model
    if args.verbose:
        print("Training the model")
    train(
        classification_model,
        train_dataloader,
        val_dataloader,
        device,
        epochs=args.epochs,
        log_wandb=args.wandb,
    )

    # Test the model
    if args.verbose:
        print("Testing the model")
    accuracy = evaluate(classification_model, test_dataloader, device)
    print("Test accuracy: %s" % (accuracy,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intent classifier training and evaluation"
    )
    parser.add_argument(
        "--embedding",
        type=str,
        default="fasttext",
        help="Name of the embedding to use (case insensitive). Must be one of : %s"
        % (list(models.keys())),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mrda",
        help="Name of the silicone dataset to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Size of batches for training",
    )
    parser.add_argument(
        "--classification_model",
        type=str,
        default="lstm",
        help="Name of the classification model to use (case insensitive). Must be one of : %s"
        % (list(layers.keys())),
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="Whether to print informations about the process",
    )
    parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        help="Whether to use weight and biases for logging the training and evaluation",
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs to train for"
    )
    parser.add_argument("--run_name", type=str, default=None, help="Name of the run")
    parser.add_argument("--run_group", type=str, default=None, help="Group of the run")
    args = parser.parse_args()

    main(args)
