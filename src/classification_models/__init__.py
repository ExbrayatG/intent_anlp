from .one_layer_mlp import OneLayerMLP
from .zero_layer_mlp import ZeroLayerMLP
from .lstm import LSTM
from .cnnlstm import CNNLSTM
from .bilstm import BiLSTM
from .cnn import CNN

layers = {
    "one_layer_mlp": OneLayerMLP,
    "zero_layer_mlp": ZeroLayerMLP,
    "lstm": LSTM,
    "cnnlstm": CNNLSTM,
    "bilstm": BiLSTM,
    "cnn": CNN,
}


def get_classification_model_class(layer_name):
    try:
        layer_class = layers[layer_name.lower()]
    except KeyError as e:
        print(
            "No classification model found for the specified layer name. Model must be one of : %s (case insensitive)"
            % (list(layers.keys()),)
        )
        raise e
    return layer_class
