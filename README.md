# Intent Classifier

This is a simple intent classifier using embeddings and classification models to perform intent recognition on various datasets. It uses PyTorch as the main framework and optionally integrates with Weights & Biases (wandb) for experiment tracking.
## Requirements
- Python 3.6+
- PyTorch
- WandB (optional)
## Installation
1. Clone this repository.
2. Install the dependencies using pip:

```bash

pip install -r requirements.txt
```

 
1. (Optional) Set up a WandB account and log in with the `wandb` CLI if you want to use experiment tracking.
## Usage

You can train and evaluate the intent classifier using `main.py`. The available command line arguments are: 
- `--embedding`: The embedding to use for input features. Must be one of: `fasttext` (default), `<other options>`. 
- `--dataset`: The name of the dataset to use. Default: `mrda`. 
- `--batch_size`: The batch size for training. Default: 16. 
- `--classification_model`: The classification model to use. Must be one of: `one_layer_mlp` (default), `<other options>`. 
- `--verbose`: Whether to print information about the process. Default: `True`. 
- `--wandb`: Whether to use Weights & Biases for logging the training and evaluation. Default: `False`. 
- `--epochs`: The number of epochs to train for. Default: 10. 
- `--run_name`: The name of the run. Default: `None`. 
- `--run_group`: The group of the run. Default: `None`.
### Example

To train and evaluate an intent classifier using FastText embeddings, the MRDA dataset, a batch size of 16, a one-layer MLP classifier, and 10 training epochs:

```bash

python main.py --embedding fasttext --dataset mrda --batch_size 16 --classification_model one_layer_mlp --epochs 10
```


### Experiment Tracking with Weights & Biases

To use Weights & Biases for experiment tracking, set the `--wandb` flag to `True`:

```bash

python main.py --embedding fasttext --dataset mrda --batch_size 16 --classification_model one_layer_mlp --epochs 10 --wandb True
```



You can also set the run name and run group for better organization in Weights & Biases:

```bash

python main.py --embedding fasttext --dataset mrda --batch_size 16 --classification_model one_layer_mlp --epochs 10 --wandb True --run_name "My Run" --run_group "My Group"
```


## Datasets

This project supports various datasets. To use a different dataset, change the `--dataset` argument.