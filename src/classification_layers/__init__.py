from .one_layer_mlp import OneLayerMLP

layers = {"one_layer_mlp": OneLayerMLP}


def get_layer_class(layer_name):
    try:
        layer_class = layers[layer_name.lower()]
    except KeyError as e:
        print("No layer found for the specified layer name")
        raise e
    return layer_class
