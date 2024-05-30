def normalize_weights(weights: list[float]) -> list[float]:
    total = sum(weights)
    return [weight / total for weight in weights]
