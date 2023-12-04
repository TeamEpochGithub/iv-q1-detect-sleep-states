from enum import Enum, auto


class EarlyStoppingMetric(Enum):
    """Enum class for the early stopping metric."""
    DISABLED = auto()
    VALIDATION_LOSS = auto()
    VALIDATION_SCORE = auto()
