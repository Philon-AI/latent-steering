

def split_param_groups(model, lr, weight_decay):
    weights, biases_and_norms = [], []

    for param in model.parameters():
        if not param.requires_grad:
            continue

        if param.ndim <= 1:
            biases_and_norms.append(param)
        else:
            weights.append(param)
    
    return [
        {"params": weights, "lr": lr, "weight_decay": weight_decay, "name": "weights"},
        {"params": biases_and_norms, "lr": lr, "weight_decay": 0.0, "name": "biases_and_norms"},
    ]
