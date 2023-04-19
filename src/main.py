from data import load_dataset, create_dataloaders
from ptokenizers import tokenize, tokenizers
import argparse
import torch
from models import get_model_class, models
from classification_layers import get_layer_class, layers
from utils import train, evaluate
from config import CLASSIFICATION_LAYERS, MODELS, TOKENIZERS
import wandb


def main(args):
    wandb_config = {
        "model": args.model,
        "classification_layer": args.classification_layer,
        "dataset": args.dataset,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "classification_layer_config": CLASSIFICATION_LAYERS.get(
            args.classification_layer, {}
        ),
        "model_config": MODELS.get(args.model, {}),
        "tokenizer_config": TOKENIZERS.get(args.model, {}),
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

    class BaseModel(torch.nn.Module):
        def __init__(self, dataset) -> None:
            self.train_dataloder = None
            self.validation_data = None
            # Tokenize
            # Put in dataloader
            # Create the embedding model
            # Create the classification model
            pass

        def forward(self, x):
            # Compute the embedding
            # Classify
            # Return logit
            pass

    model = BaseModel(dataset)

    # Train the model
    if args.verbose:
        print("Training the model")
    train(
        model,
        model.train_dataloader,
        model.val_dataloader,
        device,
        epochs=args.epochs,
        log_wandb=args.wandb,
    )

    # Test the model
    if args.verbose:
        print("Testing the model")
    accuracy = evaluate(model, model.test_dataloader, device)
    print("Test accuracy: %s" % (accuracy,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intent classifier training and evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert",
        help="Name of the model to use (case insensitive). Must be one of : %s"
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
        "--classification_layer",
        type=str,
        default="one_layer_mlp",
        help="Name of the classification layer to use (case insensitive). Must be one of : %s"
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
