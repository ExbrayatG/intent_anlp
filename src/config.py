CLASSIFICATION_MODELS = {
    "one_layer_mlp": {"hidden_dim": 300},
    "zero_layer_mlp": {},
    "lstm": {"hidden_dim": 300},
    "cnnlstm": {"hidden_dim": 300, "num_filters": 100, "kernel_size": 3},
    "bilstm": {"hidden_dim": 300},
    "cnn": {"num_filters": 100, "kernel_size": 3},
    "cnn2mlp": {"num_filters": 100, "kernel_size": 3, "hidden_dim": 300},
}
EMBEDDINGS = {}
