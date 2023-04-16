# This module can't be named "tokenizers" because it messes with imports from transformers

from .distil_bert import DistilBERTTokenizer

tokenizers = {"distilbert": DistilBERTTokenizer}


def tokenize(tokenizer_name, dataset):
    try:
        tokenizer_class = tokenizers[tokenizer_name.lower()]
    except KeyError as e:
        print("No tokenizer found for the specified model name")
        raise e
    tokenizer = tokenizer_class()
    tokenized_dataset_train = tokenizer.encode(dataset["train"]["Utterance"])
    tokenized_dataset_validation = tokenizer.encode(dataset["validation"]["Utterance"])
    tokenized_dataset_test = tokenizer.encode(dataset["test"]["Utterance"])
    return (
        tokenized_dataset_train,
        tokenized_dataset_validation,
        tokenized_dataset_test,
    )
