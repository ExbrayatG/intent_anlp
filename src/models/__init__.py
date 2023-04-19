from .distil_bert import DistilBERTModel
from .fasttext_m import FastTextModel

models = {"distilbert": DistilBERTModel, "fasttext": FastTextModel}


def get_model_class(model_name):
    try:
        model_class = models[model_name.lower()]
    except KeyError as e:
        print(
            "No model found for the specified model name. Model must be one of : %s (case insensitive)"
            % (list(models.keys()),)
        )
        raise e
    return model_class
