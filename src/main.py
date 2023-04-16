from data import load_dataset, create_dataloaders
from ptokenizers import tokenize, tokenizers
import argparse
import torch
from models import get_model_class, models
from classification_layers import get_layer_class, layers
from utils import train, evaluate


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.verbose:
        print("Running on device: ", device)

    # Load the data
    if args.verbose:
        print("Loading dataset")
    dataset = load_dataset(args.dataset)

    # Tokenize
    if args.verbose:
        print("Tokenizing dataset")
    (
        tokenized_dataset_train,
        tokenized_dataset_validation,
        tokenized_dataset_test,
    ) = tokenize(args.model, dataset)

    # Create dataloaders
    if args.verbose:
        print("Creating dataloaders")
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
        tokenized_dataset_train,
        tokenized_dataset_validation,
        tokenized_dataset_test,
        dataset,
    )

    # Create model
    if args.verbose:
        print("Creating model")
    model_class = get_model_class(args.model)
    model = model_class()

    # Add classification layer
    if args.verbose:
        print("Adding classification layer")
    classification_layer_class = get_layer_class(args.classification_layer)
    num_labels = len(dataset["train"].features["Label"].names)
    classification_layer = classification_layer_class(
        D_in=model.output_dim, H=300, D_out=num_labels  # TODO: add H to config
    )

    model.classifier = classification_layer

    # Train the model
    if args.verbose:
        print("Training the model")
    train(model, train_dataloader, val_dataloader, device, epochs=2)

    # Test the model
    if args.verbose:
        print("Testing the model")
    accuracy = evaluate(model, test_dataloader, device)
    print("Test accuracy: %s" % (accuracy,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Intent classifier training and evaluation"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="fasttext",
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
    args = parser.parse_args()

    main(args)
