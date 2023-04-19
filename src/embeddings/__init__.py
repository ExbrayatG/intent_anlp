from .distilbert_layer import DistilBertLayer
from .fasttext_layer import FastTextLayer

models = {"distilbert": DistilBertLayer, "fasttext": FastTextLayer}


def get_embedding_class(model_name):
    try:
        model_class = models[model_name.lower()]
    except KeyError as e:
        print(
            "No embedding found for the specified embedding name. Embedding must be one of : %s (case insensitive)"
            % (list(models.keys()),)
        )
        raise e
    return model_class
