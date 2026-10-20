from adapter import LinearWithAdapter

def add_adapter(model , rank , dropout):
    encoder = model.base_model.encoder
    for layer in encoder.layer:
        layer.attention.output.dense = LinearWithAdapter(base=layer.attention.output.dense,rank=rank,dropout=dropout)
        layer.output.dense = LinearWithAdapter(layer.output.dense,rank=rank,dropout=dropout)

    return model
 
def freeze_for_adapter_tuning(model, train_layernorm=True):
    """Freeze everything except adapters, classifier head (and LayerNorms)."""
    for name, p in model.named_parameters():
        trainable = (
            ".adapter." in name
            or name.startswith("classifier.")
            or (train_layernorm and "LayerNorm" in name)
        )
        p.requires_grad = trainable
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return train, total


def adapter_state_dict(model):
    """Only the weights we trained -> tiny checkpoint (a few MB)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items()
            if ".adapter." in k or k.startswith("classifier.") or "LayerNorm" in k}


def load_adapter_state_dict(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected[:5]}"
    return model
