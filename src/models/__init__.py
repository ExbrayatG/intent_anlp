from .distil_bert import DistilBERTModel

models = {"distilbert": DistilBERTModel}


def get_model_class(model_name):
    try:
        model_class = models[model_name.lower()]
    except KeyError as e:
        print("No model found for the specified model name")
        raise e
    return model_class
