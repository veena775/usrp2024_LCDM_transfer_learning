def dynamic_model_optional_fixed_final(trial, input_size, output_size, final_hidden_layer_size=None, max_layers=3, max_neurons_layers=500):
    # define the tuple containing the different layers
    layers = []

    # get the number of hidden layers
    n_layers = trial.suggest_int("n_layers", 1, max_layers)

    # get the hidden layers
    in_features = input_size
    for i in range(n_layers - (1 if final_hidden_layer_size else 0)): 
        out_features = trial.suggest_int(f"n_units_l{i}", 4, max_neurons_layers)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.LeakyReLU(0.2))
        p = trial.suggest_float(f"dropout_l{i}", 0.2, 0.8)
        layers.append(nn.Dropout(p))
        in_features = out_features

    # Add the final hidden layer if final_hidden_layer_size is specified
    if final_hidden_layer_size:
        layers.append(nn.Linear(in_features, final_hidden_layer_size))
        layers.append(nn.LeakyReLU(0.2))
        p = trial.suggest_float("dropout_final_hidden", 0.2, 0.8)
        layers.append(nn.Dropout(p))
        in_features = final_hidden_layer_size

    # Add the output layer
    layers.append(nn.Linear(in_features, output_size))
    layers.append(nn.Sigmoid())

    # return the model
    return nn.Sequential(*layers)

