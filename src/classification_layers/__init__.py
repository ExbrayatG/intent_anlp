from .one_layer_mlp import OneLayerMLP
from .zero_layer_mlp import ZeroLayerMLP

layers = {"one_layer_mlp": OneLayerMLP, "zero_layer_mlp": ZeroLayerMLP}


def get_layer_class(layer_name):
    try:
        layer_class = layers[layer_name.lower()]
    except KeyError as e:
        print(
            "No classification layer found for the specified layer name. Layer must be one of : %s (case insensitive)"
            % (list(layers.keys()),)
        )
        raise e
    return layer_class
