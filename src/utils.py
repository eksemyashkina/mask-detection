class EMA:
    def __init__(self, alpha: float = 0.9) -> None:
        self.value = None
        self.alpha = alpha
    
    def __call__(self, value: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * value
        return self.value